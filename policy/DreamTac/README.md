# DreamTac — closed-loop evaluation in Isaac Sim

Runs a Dream-Tac (Cosmos-Policy 2B video2world) checkpoint through the ordinary
UniVTAC evaluation entry point. Unlike ACT and ViTAL, the model does not live in
the simulator's process: it runs in the Dream-Tac uv environment behind an HTTP
server on the same workstation. This package is the client.

```
thasos, terminal 2 (conda UniVTAC)           thasos, terminal 1 (uv, ~/proj/Dream-Tac)
┌──────────────────────────────┐             ┌──────────────────────────────┐
│ scripts/eval_policy.py       │             │ franka_server.py       :5001 │
│  └ policy/DreamTac           │  POST /infer│  └ cosmos_policy 2B + Wan VAE│
│     ├ encode_obs  ───────────┼────────────►│     get_action(...)          │
│     ├ IKServo (curobo)  ◄────┼─────────────┤     20 × [xyz, rpy, gripper] │
│     └ take_action('qpos')    │   actions   └──────────────────────────────┘
└──────────────────────────────┘
```

Nothing under `envs/` changes.

---

## Runbook

One server serves **one checkpoint of one task** (its weights and that task's
normalisation statistics), so the server is started per task and the client
refuses to run if the two disagree. `ACT`'s `EP_NUM` / `TRAIN_CONFIG` do not
apply here; the server environment picks the weights.

### 0. Stage a checkpoint (once per task / iteration)

```bash
DREAMTAC_TASK=insert_hole DREAMTAC_ITERS=iter_000004000 bash policy/DreamTac/tools/copy_checkpoint.sh
```

Copies `model/`, the stats and the T5 cache from `~/cluster_backup/models/Dream-Tac`
(override with `DREAMTAC_SRC`) into `~/models/dream-tac`. insert_hole
`iter_000004000` is already staged.

### 1. Terminal 1 — start the inference server

```bash
DREAMTAC_TASK=insert_hole DREAMTAC_ITER=iter_000004000 bash policy/DreamTac/tools/serve_local.sh
#   -> use_tactile=True (model state_t=12)
#   -> * Running on http://127.0.0.1:5001
```

Both variables default to the values shown. `DREAMTAC_TASK` is the training task
set as Dream-Tac spells it — a single task, or several joined by `__` in sorted
order for a multi-task run.

### 2. Terminal 2 — verify and evaluate

```bash
conda activate UniVTAC

# verify the server is up and holds the right checkpoint
curl -s http://127.0.0.1:5001/health   # {"model_loaded":true,...,"use_tactile":true}
curl -s http://127.0.0.1:5001/info     # ckpt_path, chunk_size

# optional: client encoding vs ground truth, no simulator
python policy/DreamTac/tools/client_check.py \
    --hdf5 data/insert_hole/clean/7.hdf5 \
    --reference ~/models/dream-tac/runs/insert_hole/offline_check_insert_hole_iter_000004000.json

# smoke test: one episode
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py insert_hole clean DreamTac/deploy --total_num 1

# full evaluation (100 episodes)
bash eval_policy.sh insert_hole clean DreamTac/deploy 0
```

Results land in `eval_result/DreamTac/<task>/deploy/<run>-<iter>/<timestamp>/`,
e.g. `.../insert_hole-iter_000004000/...`; the run tag is read from the server's
`/info`.

Use `eval_policy.sh`, not `parallel_eval.sh`: the workers would share one
server, whose requests are not designed to run concurrently, and each extra
Isaac Sim instance competes for the same GPU as the model.

### Point the client somewhere else

`deploy.yml` holds `server_url`; `DREAMTAC_SERVER` overrides it without editing
the file.

---

## Status

Verified end to end:

| what | how | result |
|---|---|---|
| checkpoint quality | `offline_check.py`, held-out episode 7, in-process | pos **1.69 mm**, rot **0.95 mrad**, grip 5.5 µm |
| client encoding | `client_check.py`, same episode, through the server | pos **1.71 mm**, and **0.05 mm** from the in-process prediction |
| proprio transform | every frame t = 0…90 vs the dataset's episode-level computation | max diff **1e-8** |
| server | `/health`, `/info` | `use_tactile=True`, `state_t=12` |
| memory | `torch.cuda.max_memory_allocated()` | **5.04 GB** steady, 8.10 GB transient at load |
| latency | per query, 5 denoising steps | **0.92 s** on A100-80GB, **2.9 s** on A40 |

**Not yet exercised:** everything downstream of `encode_obs` inside a live
simulator — `IKServo`, the qpos execution path, and whether the predicted poses
actually drive the arm sensibly. Two attempts stopped before reaching any of it
(see *GPU contention* below), so treat those parts as written-but-unproven.

---

## Files

| | |
|---|---|
| `deploy.yml` | client config: server URL, chunk execution, IK, debugging |
| `deploy_policy.py` | `Policy`: observation encoding, HTTP, action execution; `get_run_tag` |
| `ik_servo.py` | absolute EE pose → joint targets, via a world-free curobo `IKSolver` |
| `tools/copy_checkpoint.sh` | stage a checkpoint + stats + T5 from a training-layout directory |
| `tools/serve_local.sh` | run the inference server on `:5001` |
| `tools/client_check.py` | client encoding vs. ground truth, no simulator |
| `tools/offline_check.py` | runs the model in-process on a held-out episode |
| `tools/serve_univtac.sbatch`, `tools/offline_check.sbatch`, `tools/tunnel.sh` | cluster-only, no longer used (alex access ended) |

The server uses the uv environment in `~/proj/Dream-Tac/cosmos_policy/.venv`
(torch 2.6, flash-attn, natten and transformer_engine all run on the 4090), not the
UniVTAC conda env. Verified with `client_check.py`: 1.70 mm / 0.95 mrad against
ground truth, within 0.05 mm of `offline_check.py`, ~1.9 s per query, ~5 GB VRAM.

---

## What the checkpoint is, and what it is not

A fine-tuned checkpoint holds **only the fine-tuned network weights**. Reading
`iter_000004000/model/.metadata` gives:

```
top-level prefixes: {'net': 687}      # 687 tensors, all of them net.*
```

No VAE, no text encoder, no base weights. `load_model_from_checkpoint` rebuilds
the model skeleton from the experiment config *in code* and then loads those 687
tensors into it, so three further things are needed for the weights to mean
anything:

| | size | why |
|---|---|---|
| Dream-Tac source | 41 MB | builds the skeleton the weights load into |
| `tokenizer.pth` (Wan2.1 VAE) | 485 MB | `config.yaml:410` resolves it through an `hf://` URI at instantiation |
| stats `.json` + `t5_embeddings.pkl` | 8 MB | proprio/action normalisation, text conditioning |
| `model-480p-16fps.pt` (Cosmos base) | 3.7 GB | must *exist* under `HF_HOME`, see below |

The base weights are never loaded — the 687 tensors replace them, and
`model_loader.py:88` overrides `config.checkpoint.load_path` with our checkpoint
path. But `cosmos_policy_experiment_configs.py` calls
`get_checkpoint_path("hf://.../model-480p-16fps.pt")` at **import time**, which
runs `hf_hub_download`; with `HF_HUB_OFFLINE=1` and no cached file the server dies
with `LocalEntryNotFoundError` before touching our checkpoint. The repo is gated,
so without an HF token copy the blob from the cluster's `$WORK/hf` (its file name
is its sha256). The T5 encoder is not needed — the embeddings are precomputed.

There is also **no "best" checkpoint** to choose. Validation is disabled in every
Dream-Tac experiment (the inherited `validation_step` is an upstream stub
expecting keys no policy dataset emits), so the 20 saved iterations are plain
periodic snapshots and `latest_checkpoint.txt` → `iter_000004000` is simply the
last one. Pull a second one by hand to compare, and let `offline_check.py` say
which is better.

Inference reads only the `model/` sub-directory (`optim/`, `scheduler/`,
`trainer/` exist to resume training), so one checkpoint is **3.7 GB, not 7.3**.

On thasos this has already been done, and things landed at:

```
~/proj/Dream-Tac/                                       source (git repo)
~/models/hf/                                            HF_HOME, tokenizer.pth + model-480p-16fps.pt
~/models/dream-tac/cache/univtac/                       stats + t5_embeddings.pkl
~/models/dream-tac/runs/insert_hole/checkpoints/
    iter_000004000/model/                               the 687 fine-tuned tensors
```

`tools/copy_checkpoint.sh` repeats this for another task or iteration from
`~/cluster_backup/models/Dream-Tac` (the base weights are per-machine, not
per-task, so it does not fetch them). `tools/serve_local.sh` reads these files.
The insert_hole `iter_000004000` copy here is md5-identical to the backup's
checkpoint `.metadata`, stats, T5 cache and split manifest.

---

## Re-checking a checkpoint offline

Worth doing whenever the checkpoint, the statistics or the preprocessing change.
It isolates "are the weights and the pipeline right?" from "is the closed loop
right?", and reports peak VRAM.

With the server running for the task, `client_check.py` does this through the
same client path the simulator uses (to compare iterations, stage both and serve
each in turn):

```bash
python policy/DreamTac/tools/client_check.py --hdf5 data/<task>/clean/<held-out>.hdf5
```

Use a held-out episode: they are listed per task under `tasks.<task>.val` in
`~/models/dream-tac/cache/univtac/split_manifest.json`. For `insert_hole` they are
`{51, 7, 54, 39, 46}.hdf5`. If the
predicted chunk does not track ground truth to a couple of millimetres, stop
there; nothing downstream will work.

Position error does not grow across the chunk — 2.06 mm at step 0 versus 2.29 mm
at step 19, flat in between — which is the evidence for replaying the full
20-step chunk open-loop rather than re-querying part way through.

---

## Decisions worth knowing about

**Proprio needs no forward kinematics.** `observation['embodiment']['ee']` is
Isaac Lab's `panda_hand` link pose relative to the articulation root, which is
exactly what `robot_manager.get_ee_pose()` returns and exactly what was recorded
as `embodiment/ee` during collection. The client only converts quaternion (wxyz)
to Euler (sxyz) and reproduces the dataset's roll handling.

**Roll is unwrapped online.** The dataset takes roll modulo 2π and then
`np.unwrap`s the episode, because the downward-pointing gripper sits exactly on
`arctan2`'s branch cut. `np.unwrap` is causal, so keeping one previous value
reproduces it exactly; `Policy.reset()` clears that state per episode. This is
what `client_check.py` verifies to 1e-8. Skipping it makes roll flip by 2π
between frames and the normalised proprio explode.

**Frames already agree.** curobo's `base_link` is `panda_link0`, which is also
the articulation root that `get_ee_pose()` subtracts, and the robot spawns at the
origin with identity rotation (no `pos=` override anywhere in the config chain).
Model output, `embodiment/ee` and the IK goal are all in the same frame, so no
transform is applied beyond the planner's `frame_bias` (`[0,0,0]` for the Franka).

**IK, not motion planning.** `take_action(..., 'ee')` would route each action
through `CuroboPlanner.plan_path` — a full trajectory plan per action, twenty per
chunk — and a single planning failure latches `plan_success = False`, silently
voiding every remaining action in the episode. Instead the client solves IK
itself and uses the plain `'qpos'` path. See `ik_servo.py` for why the solver is
built without a collision world and why every solve is seeded with the current
configuration.

**Images reproduce the training round trip.** Collection wrote frames with
`cv2.imencode(".jpg", rgb)` and the Dream-Tac adapter reads them with
`cv2.imdecode`; both assume BGR, so the swaps cancel and the simulator's native
RGB order survives — but the JPEG quantisation happens with R and B exchanged, so
it cannot be reproduced by encoding through PIL. `_to_training_view` runs the same
cv2 calls, then `INTER_AREA` to 224, and ships the result as **PNG** so the
transport adds no second lossy pass. Measured cost of the whole client route
versus feeding the dataset arrays in-process: 0.05 mm.

---

## Tuning

**`sim_steps_per_action` (default 2).** Data was recorded every
`save_frequency = 2` sim steps, so one recorded transition `ee[t] → ee[t+1]`
spans two steps, while `take_action` advances one. The default holds each target
for two steps to restore the training timebase. (The ACT baseline uses one, and
therefore replays its trajectories at double speed.) Set it to 1 to match ACT.

**`actions_per_chunk` (default: all 20).** The paper quotes 5 Hz at 1109 ms per
query with `H=20`, which only reconciles if a partial chunk is executed —
executing all 20 open-loop would be ~18 Hz. The offline check shows no error
growth across the chunk, so full replay is the default; try 5–8 if the arm drifts
away from the observation it planned from.

**Latency.** Halving the denoising steps from 10 to 5 only moved latency from the
paper's quoted 1109 ms to 920 ms, so the fixed overhead — VAE-encoding the input
sequence and decoding predicted future frames — dominates, not the denoising.
Those future frames are never read by closed-loop control. Passing
`generate_future_state_and_value_in_parallel=False` in `franka_server.py:248`
should cut most of it. Untested.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'flask_cors'`.** The uv environment has
`flask` but not `flask_cors`. Rather than mutate an environment that training
jobs share (and that `uv sync` would reset), `franka_server.py` was patched to
make the import optional — CORS only matters for browser clients. The patch is
applied in `~/proj/Dream-Tac`.

**`PxgCudaDeviceMemoryAllocator failed to allocate memory 268435456 bytes` →
`Unable to create scene` → `Articulation ... _root_physx_view`.** GPU contention,
not a code fault: everything after the first line is a cascade from PhysX never
creating a physics scene. thasos's single RTX 4090 is **shared with other users**;
check before blaming anything else:

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

Isaac Sim plus the UIPC tactile simulation needs more than the ~5.5 GB that was
free during both attempts (it reached 4351 MiB and then failed on the next 256 MiB
PhysX heap chunk). The exact requirement is unmeasured — that needs a moment with
the card to itself.

The inference server now shares that card too (~5 GB steady, ~8 GB while
loading), so start it first, let it finish loading, and check that enough is
left for the simulator before launching the evaluation.

**Client cannot reach the server / "Server holds a checkpoint trained on ...".**
`Policy.__init__` asks the server for `/health` and `/info` before the simulator
starts. It fails if nothing is listening on `server_url`, and also if the
server's checkpoint was trained on a task set that does not include the task
under evaluation (e.g. a server left running from a previous task). Restart
`serve_local.sh` with the matching `DREAMTAC_TASK`.
