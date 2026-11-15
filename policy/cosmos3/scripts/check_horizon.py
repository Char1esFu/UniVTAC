"""Checks for decoupling Cosmos3's video clip from the action chunk (patches/0001-video-horizon.patch).

vae     Reconstruction PSNR of the frozen Wan2.2 VAE on UniVTAC clips whose 17 frames are `stride`
        dataset frames apart, per camera region of the canvas, next to the PSNR of simply repeating the
        condition frame (how much the scene changes over the clip). If the VAE cannot represent clips at a
        stride, the world model cannot predict them either.
policy  Builds the patched policy for several (video_stride, video_latent_frames, chunk_size) settings,
        runs one training forward on a dataset batch and samples one chunk, and prints the M-RoPE time of
        every latent frame and action in dataset frames.

    .venv/bin/python check_horizon.py vae --root $DATASET_ROOT
    .venv/bin/python check_horizon.py policy --root $DATASET_ROOT
"""

import argparse
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from dataclasses import asdict
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.cosmos3.configuration_cosmos3 import Cosmos3EdgeConfig, Wan2pt2VAEConfig
from lerobot.policies.cosmos3.processor_cosmos3 import Cosmos3VideoCanvasProcessorStep
from lerobot.policies.factory import make_policy, make_pre_post_processors

HEAD, WRIST, TAC = "observation.images.head", "observation.images.wrist", "observation.images.tactile_ll_marker"
LAYOUTS = {"2v": [HEAD, WRIST], "3v": [HEAD, WRIST, TAC]}
CHECKPOINT = "lerobot/Cosmos3-Edge"


def load_vae(device):
    """The Wan2.2 VAE of the Cosmos3-Edge checkpoint, float32."""
    vae = AutoencoderKLWan.from_config(asdict(Wan2pt2VAEConfig()))
    path = hf_hub_download(CHECKPOINT, "model.safetensors")
    state = {}
    with safe_open(path, "pt") as f:
        for key in f.keys():
            if key.startswith("vae.") or ".vae." in key:
                state[key.split("vae.", 1)[1]] = f.get_tensor(key).float()
    vae.load_state_dict(state, strict=True)
    return vae.to(device).eval()


def regions(layout, ch, cw):
    """(name, rows, cols) of every camera in the fitted canvas content."""
    if layout == "2v":
        return [("head", slice(0, ch), slice(0, cw // 2)), ("wrist", slice(0, ch), slice(cw // 2, cw))]
    top = ch * 2 // 3
    return [
        ("head", slice(0, top), slice(0, cw)),
        ("wrist", slice(top, ch), slice(0, cw // 2)),
        ("tactile_ll", slice(top, ch), slice(cw // 2, cw)),
    ]


def psnr(mse):
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


@torch.no_grad()
def check_vae(args):
    device = "cuda"
    ds = LeRobotDataset(args.repo_id, root=args.root)
    starts = np.asarray(ds.meta.episodes["dataset_from_index"])
    ends = np.asarray(ds.meta.episodes["dataset_to_index"])
    rng = np.random.default_rng(0)
    span = 16 * max(args.strides)
    windows = []
    while len(windows) < args.windows:
        ep = rng.integers(len(starts))
        if ends[ep] - starts[ep] > span:
            windows.append(int(rng.integers(starts[ep], ends[ep] - span)))
    cache = {}

    def frame(idx, key):
        if idx not in cache:
            item = ds[idx]
            cache[idx] = {k: item[k] for k in (HEAD, WRIST, TAC)}
        return cache[idx][key]

    vae = load_vae(device)
    step = Cosmos3VideoCanvasProcessorStep(camera_keys=[], view_size=(256, 256), fps=ds.meta.fps, chunk_size=16)
    print(f"{len(windows)} windows, 17 frames each (5 latent frames), dataset fps {ds.meta.fps}")
    print(f"{'layout':6} {'stride':>6} {'clip (s)':>8} {'region':>10} {'VAE PSNR':>9} {'repeat t0':>9} {'VAE t0':>7}")
    for layout, cams in LAYOUTS.items():
        for stride in args.strides:
            sums = {}
            t0 = time.time()
            for b in range(0, len(windows), args.batch):
                group = windows[b : b + args.batch]
                views = []
                for cam in cams:
                    clip = torch.stack([torch.stack([frame(w + stride * i, cam) for i in range(17)]) for w in group])
                    views.append(step.prepare_view(clip.float(), (256, 256)))
                frames = (step.compose(views) * 255.0).clamp(0, 255).to(torch.uint8)  # [B,T,3,H,W]
                canvas, (ch, cw) = step.fit_canvas(frames)
                x = canvas.permute(0, 2, 1, 3, 4).float().to(device) / 127.5 - 1.0  # [B,3,T,H,W]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = vae.encode(x).latent_dist.mode()
                    y = vae.decode(z).sample
                ref = (x[..., :ch, :cw].float() + 1) / 2
                rec = ((y[..., :ch, :cw].float().clamp(-1, 1)) + 1) / 2
                for name, rows, cols in regions(layout, ch, cw):
                    r, p = ref[..., rows, cols], rec[..., rows, cols]
                    err = ((r - p) ** 2).mean(dim=(1, 3, 4))  # [B,T]
                    rep = ((r - r[:, :, :1]) ** 2).mean(dim=(1, 3, 4))
                    acc = sums.setdefault(name, [0.0, 0.0, 0.0, 0])
                    acc[0] += err[:, 1:].mean(1).sum().item()
                    acc[1] += rep[:, 1:].mean(1).sum().item()
                    acc[2] += err[:, 0].sum().item()
                    acc[3] += len(group)
            for name, (e, r, e0, n) in sums.items():
                print(f"{layout:6} {stride:>6} {16 * stride / ds.meta.fps:>8.2f} {name:>10} "
                      f"{psnr(e / n):>9.2f} {psnr(r / n):>9.2f} {psnr(e0 / n):>7.2f}")
            print(f"       ({time.time() - t0:.0f}s, canvas content {ch}x{cw})", flush=True)


def check_policy(args):
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    cams = LAYOUTS["2v"]
    for stride, latent, chunk in args.settings:
        print(f"\n===== video_stride={stride} video_latent_frames={latent} chunk_size={chunk}", flush=True)
        cfg = Cosmos3EdgeConfig(
            pretrained_path=CHECKPOINT, load_action_modules=False, chunk_size=chunk, n_action_steps=chunk,
            video_stride=stride, video_latent_frames=latent, num_inference_steps=10, device="cuda",
            input_features={c: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for c in cams},
            view_description="The left half shows the third-person head camera; the right half shows the "
            "wrist-mounted camera.",
        )
        policy = make_policy(cfg, ds_meta=meta)
        pre, post = make_pre_post_processors(cfg, dataset_stats=meta.stats)
        delta = resolve_delta_timestamps(cfg, meta)
        delta = {k: v for k, v in delta.items() if not k.startswith("observation.image") or k in cfg.input_features}
        print("clip frame offsets:", cfg.video_frame_offsets, "| actions:", chunk)
        ds = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta)
        starts = np.asarray(ds.meta.episodes["dataset_from_index"])
        items = [ds[int(s) + 10] for s in starts[:2]]
        raw = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        raw["task"] = [it["task"] for it in items]

        captured = {}
        rope = policy.model.network.get_rope_index

        def capture(*a, **kw):
            out = rope(*a, **kw)
            captured.update(out=out, grid=a[1], num_actions=kw.get("num_actions", 0))
            return out

        policy.model.network.get_rope_index = capture
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = pre(dict(raw))
            print("pixel_values_videos_gen", tuple(batch["pixel_values_videos_gen"].shape),
                  "| latent grid", batch["video_grid_thw_gen"][0].tolist(),
                  "| s per latent frame", round(float(batch["second_per_grid_ts_gen"][0]), 4))
            print("prompt", batch.get("prompt", [""])[0] if "prompt" in batch else "")
            loss, info = policy.forward(batch)
            print(f"train forward: loss {loss.item():.3f} video {float(info['video_loss']):.3f} "
                  f"action {float(info['action_loss']):.3f}")

            infer = {k: (v[:, 0] if k in cfg.input_features else v) for k, v in raw.items() if k != "action"}
            t0 = time.time()
            actions = post(policy.predict_action_chunk(pre(infer)))
            torch.cuda.synchronize()
            print(f"sampled chunk {tuple(actions.shape)} finite={bool(torch.isfinite(actions).all())} "
                  f"in {time.time() - t0:.2f}s (10 steps, batch 2)")

        pos = captured["out"][0, 0].float()  # temporal axis, first sample
        t_lat, h, w = captured["grid"][0].tolist()
        n_act = captured["num_actions"]
        n_vid = t_lat * (h // cfg.latent_patch_size) * (w // cfg.latent_patch_size)
        video_t, action_t = pos[-n_act - n_vid : -n_act], pos[-n_act:]
        start = video_t[0]
        per_frame = cfg.base_fps / (cfg.vae.scale_factor_temporal or 1) / meta.fps  # rope units per dataset frame
        lat = ((video_t.view(t_lat, -1)[:, 0] - start) / per_frame).tolist()
        act = ((action_t - start) / per_frame).tolist()
        print("latent frames at dataset frame:", [round(x, 2) for x in lat])
        print("actions at dataset frame:", [round(x, 2) for x in act[:6]], "...", [round(x, 2) for x in act[-2:]])
        del policy
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["vae", "policy"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", default="local/put_bottle_in_shelf-clean-100_vitac")
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8])
    parser.add_argument("--windows", type=int, default=48)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--settings", nargs="+", default=["1:-:16", "2:-:32", "5:2:50", "2:4:16"],
        help="video_stride:video_latent_frames:chunk_size, '-' for None",
    )
    args = parser.parse_args()
    args.settings = [
        (int(s), None if k == "-" else int(k), int(c)) for s, k, c in (x.split(":") for x in args.settings)
    ]
    check_vae(args) if args.check == "vae" else check_policy(args)


if __name__ == "__main__":
    main()
