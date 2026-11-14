"""Precompute the frozen latents used by the SmolVLA world action model.

For every frame of a processed LeRobot dataset this writes, under
``<dataset root>/wam_latents/``:

* ``<camera>.npy``  (N, 64, 960) float16 -- SigLIP + connector tokens (8x8 grid)
  of each camera, computed exactly as ``SmolVLMWithExpertModel.embed_image``
  sees them in training (resize_with_pad to 512, [-1, 1]).
* ``<tactile stream>.npy``  (N, 512) float32 -- the frozen pretrained UniVTAC
  backbone feature of each tactile stream (input in [0, 1], resized to 224).
* ``frames.npz`` -- per global index: episode_index, frame_index and the
  episode's [from, to) global index range.

Rows are indexed by the dataset's global ``index``. Pooling and normalization
are applied at training time, so they can change without recomputing.

    .venv/bin/python scripts/precompute_wam_latents.py lift_bottle clean 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel  # noqa: E402
from lerobot.policies.smolvla.vitac_encoder import UniVTACEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--root", type=Path, default=None, help="dataset root (default data/local/<repo>)")
    parser.add_argument("--cameras", nargs="+", default=["observation.images.head", "observation.images.wrist"])
    parser.add_argument(
        "--tactile", nargs="*",
        default=["observation.images.tactile_ll_marker", "observation.images.tactile_lr_marker"],
        help="tactile streams; pass --tactile with no value to skip them (fine-tuned tactile "
             "encoder: its latents are computed online during training)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output dir (default <root>/wam_latents)")
    parser.add_argument("--vlm", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--vitac-ckpt", default="encoder/checkpoints/resnet18/20251128-125750/best.pth")
    parser.add_argument("--image-size", type=int, default=512, help="SmolVLA resize_imgs_with_padding")
    parser.add_argument("--tactile-size", type=int, default=224, help="SmolVLA vitac_input_size")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    repo_id = f"local/{args.task_name}-{args.task_config}-{args.expert_data_num}_vitac"
    root = args.root or ROOT / "data" / repo_id
    out_dir = args.out or root / "wam_latents"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    ds = LeRobotDataset(repo_id=repo_id, root=root)
    n = ds.num_frames
    keys = list(args.cameras) + list(args.tactile)
    missing = [k for k in keys if k not in ds.meta.features]
    if missing:
        raise KeyError(f"{missing} not in dataset features {sorted(ds.meta.features)}")

    vlm = SmolVLMWithExpertModel(model_id=args.vlm, load_vlm_weights=True, num_vlm_layers=16).to(device)
    vlm.eval()  # its train() override returns None, so no chaining
    tactile = UniVTACEncoder(pretrained_path=args.vitac_ckpt).backbone.to(device).eval() if args.tactile else None

    outs = {
        k: np.lib.format.open_memmap(out_dir / f"{k}.npy", mode="w+", dtype=np.float16, shape=(n, 64, 960))
        for k in args.cameras
    }
    outs |= {
        k: np.lib.format.open_memmap(out_dir / f"{k}.npy", mode="w+", dtype=np.float32, shape=(n, 512))
        for k in args.tactile
    }
    episode_index = np.zeros(n, dtype=np.int64)
    frame_index = np.zeros(n, dtype=np.int64)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"latents {args.task_name}"):
            idx = batch["index"].numpy()
            episode_index[idx] = batch["episode_index"].numpy()
            frame_index[idx] = batch["frame_index"].numpy()
            for k in args.cameras:
                img = batch[k].to(device, non_blocking=True)
                img = resize_with_pad(img, args.image_size, args.image_size, pad_value=0) * 2.0 - 1.0
                outs[k][idx] = vlm.embed_image(img).float().cpu().numpy().astype(np.float16)
            for k in args.tactile:
                img = batch[k].to(device, non_blocking=True)
                img = F.interpolate(img, size=(args.tactile_size,) * 2, mode="bilinear", align_corners=False)
                outs[k][idx] = tactile(img).float().cpu().numpy()

    episodes = ds.meta.episodes
    ep_from = np.asarray(episodes["dataset_from_index"], dtype=np.int64)
    ep_to = np.asarray(episodes["dataset_to_index"], dtype=np.int64)
    np.savez(
        out_dir / "frames.npz",
        episode_index=episode_index,
        frame_index=frame_index,
        episode_from=ep_from[episode_index],
        episode_to=ep_to[episode_index],
    )
    for arr in outs.values():
        arr.flush()
    (out_dir / "info.json").write_text(json.dumps({
        "num_frames": n, "fps": ds.fps, "cameras": args.cameras, "tactile": args.tactile,
        "vlm": args.vlm, "vitac_ckpt": args.vitac_ckpt,
        "image_size": args.image_size, "tactile_size": args.tactile_size,
    }, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
