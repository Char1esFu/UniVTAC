"""Mirror the Trackio runs of the Cosmos3 trainer to Weights & Biases.

The Cosmos3 lerobot tracks experiments with Trackio only (no wandb backend). Trackio writes every logged
metric to JSONL files under $HF_HOME/trackio/inbox/<writer>/, so a run can be uploaded to wandb at any time,
including the steps logged before this script first ran. Each Trackio run maps to one wandb run with a
stable id (derived from the Trackio run name, which a resumed job keeps), and only steps above the last
uploaded one are sent, so the script can be rerun or looped freely.

    uv run --no-project --with wandb python sync_wandb.py [--loop 300] [--run-filter <substring>]

Needs wandb credentials (~/.netrc from `wandb login`, or WANDB_API_KEY) and internet access (login node,
or the NHR proxy on a compute node).
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def read_records(trackio_dir: Path, project: str) -> dict[str, dict]:
    """Returns {run name: {"config": dict | None, "steps": {step: metrics}}} from the Trackio inbox."""
    runs: dict[str, dict] = {}
    for path in sorted((trackio_dir / "inbox").glob("*/*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a file still being written
                continue
            if rec.get("kind") != "metric" or rec.get("project") != project or rec.get("step") is None:
                continue
            run = runs.setdefault(rec["run"], {"config": None, "steps": {}})
            if rec.get("config") and run["config"] is None:
                run["config"] = rec["config"]
            # A resumed job relogs the steps after its checkpoint; keep the latest values per step.
            run["steps"].setdefault(int(rec["step"]), {}).update(rec["metrics"])
    return runs


def wandb_run_id(run_name: str) -> str:
    return "trackio-" + hashlib.sha1(run_name.encode()).hexdigest()[:16]


def sync_once(args) -> None:
    import wandb

    runs = read_records(args.trackio_dir, args.project)
    for name, run in sorted(runs.items()):
        if args.run_filter and args.run_filter not in name:
            continue
        # One state file per run, so loops syncing different runs never overwrite each other.
        state = args.state_dir / f"{wandb_run_id(name)}.json"
        last = json.loads(state.read_text())["step"] if state.exists() else -1
        new_steps = sorted(s for s in run["steps"] if s > last)
        if not new_steps:
            continue
        print(f"[sync_wandb] {name}: steps {new_steps[0]}..{new_steps[-1]} ({len(new_steps)} logs)", flush=True)
        wb = wandb.init(
            entity=args.entity,
            project=args.wandb_project,
            id=wandb_run_id(name),
            name=name,
            config=run["config"] or {},
            resume="allow",
            dir=args.state_dir,  # keeps the local wandb/ run files out of the working directory
            settings=wandb.Settings(
                x_disable_stats=True, console="off", silent=True, x_disable_meta=True,
            ),
        )
        for step in new_steps:
            wb.log(run["steps"][step], step=step)
        wb.finish()
        state.write_text(json.dumps({"run": name, "step": new_steps[-1]}))


def main() -> None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trackio-dir", type=Path, default=hf_home / "trackio")
    p.add_argument("--project", default="cosmos3_univtac", help="Trackio project")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "cong_fu"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "cosmos3_univtac"))
    p.add_argument("--run-filter", default=None, help="only Trackio runs whose name contains this")
    p.add_argument("--state-dir", type=Path, default=None,
                   help="last uploaded step per run (default: <trackio dir>/wandb_sync)")
    p.add_argument("--loop", type=float, default=0, help="resync every this many seconds (0: once)")
    args = p.parse_args()
    args.state_dir = args.state_dir or args.trackio_dir / "wandb_sync"
    args.state_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            sync_once(args)
        except Exception as e:  # network hiccups must not end a background loop
            if not args.loop:
                raise
            print(f"[sync_wandb] sync failed, retrying later: {e!r}", flush=True)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
