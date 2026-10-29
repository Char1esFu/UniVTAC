# FTP1 — closed-loop evaluation in Isaac Sim

Runs the released FTP-1 UniVTAC checkpoints (`ftp1_univtac_finetune`, one model per
task) through the ordinary UniVTAC evaluation entry point. FTP-1 needs
transformers 4.53.2 with openpi's `transformers_replace` patch, which cannot live in
the `UniVTAC` conda env (transformers 5.x), so the model runs in the ftp1-policy uv
env behind an HTTP server and this package is the client.

```
terminal 2 (conda UniVTAC)                    terminal 1 (uv, ~/proj/ftp1-policy)
┌──────────────────────────────┐              ┌──────────────────────────────────┐
│ scripts/eval_policy.py       │  POST /load  │ policy/FTP1/ftp1_server.py :5002 │
│  └ policy/FTP1 (client)      ├─────────────►│  └ FTP1InferenceWrapper          │
│     ├ head/wrist/tactile/qpos│  POST /infer │     PaliGemma 2B + action expert │
│     ├ temporal ensemble ◄────┼──────────────┤     + tactile expert (T3)        │
│     └ take_action('qpos')    │  32×8 chunk  └──────────────────────────────────┘
└──────────────────────────────┘
```

Nothing under `envs/` changes. Execution mirrors `UniVTAC/scripts/eval_ftp1.py` +
`eval_ftp1.sh` in the ftp1-policy repo, which produced the published numbers.

---

## Runbook

### 0. One-time setup

- **ftp1-policy env**: `~/proj/ftp1-policy` with `uv sync`, `uv pip install -e .` and the
  `transformers_replace` copy done (see that repo's README).
- **Checkpoints** (47 GB, git-ignored), into `policy/FTP1/checkpoints/`:
  ```bash
  cd ~/proj/UniVTAC
  uv run --project ~/proj/ftp1-policy huggingface-cli download MJJJJ1064/ftp1_univtac_finetune \
      --local-dir policy/FTP1/checkpoints/ftp1_univtac_finetune
  ```
- **PaliGemma tokenizer**: openpi fetches it from GCS on first use, which fails on this
  network. Fetch it over HTTPS instead:
  ```bash
  mkdir -p ~/.cache/openpi/big_vision
  curl -fL -o ~/.cache/openpi/big_vision/paligemma_tokenizer.model \
      https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
  ```

### 1. Terminal 1 — inference server

```bash
cd ~/proj/UniVTAC
uv run --project ~/proj/ftp1-policy python policy/FTP1/ftp1_server.py
```

Prints `FTP-1 server listening on http://127.0.0.1:5002` and waits. No model is loaded
until a client calls `/load`; the server then picks the checkpoint for that task and
swaps it when the task changes, so one server covers all tasks. Add
`--preload_task insert_tube` to load at startup. Check it with
`curl http://127.0.0.1:5002/health`.

Resident GPU memory is ~8 GB, leaving the rest of a 24 GB card for Isaac Sim.

### 2. Terminal 2 — evaluation

```bash
conda activate UniVTAC
cd ~/proj/UniVTAC
bash eval_policy.sh insert_tube contact FTP1/deploy 0
#                   task        task_config  deploy yml   GPU
```

`contact` is the task config the official FTP-1 numbers used. `eval_policy.sh` only
forwards these three arguments; for seed control call the script directly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py insert_tube contact FTP1/deploy \
    --total_num 100 --start_seed 1000000 [--expert_check]
```

Default is 100 episodes from seed 1000000. Each step re-infers (~120 ms), so an
episode takes about 1 min.

### Tasks

| task | checkpoint | cameras |
|---|---|---|
| `insert_hole` | `FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999` | head |
| `insert_tube` | `FTP1_UniVTAC_insert_tube_expert_gsmall_ftp1/19999` | head + wrist |
| `lift_bottle` | `FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1/19999` | head |
| `lift_can` | `FTP1_UniVTAC_lift_can_expert_gsmall_ftp1/19999` | head + wrist |
| `pull_out_key` | `FTP1_UniVTAC_pull_out_key_expert_gsmall_ftp1/19999` | head |
| `put_bottle_in_shelf` | `FTP1_UniVTAC_put_bottle_expert_gsmall_ftp1/19999` | head |

`grasp_classify` and `insert_HDMI` have no released checkpoint; train one and set
`checkpoint_dir` (below).

### Results

`eval_result/FTP1/<task>/deploy/<run_tag>/<timestamp>/`: `log.log` (per-seed result and
running success rate), `video/<seed>_{success,failed}.mp4`, `metadata.json`. The
`run_tag` is `ftp1_univtac_finetune` for the release, or `<run>-<step>` for a custom
`checkpoint_dir`.

---

## What the client sends

All of this follows how the checkpoints were trained
(`data_processing/parse_data_module/parse_data_univtac.py` in ftp1-policy):

- **Prompt**: FTP-1 is a VLA and always takes text. The client sends the fixed
  training instruction for the task (e.g. `"insert the tube to the fixed slot."`), not
  `task.instruction`, which is `"Empty"` with `instuction_file: null`. The resolved
  prompt is written into the `Eval Config` block of `log.log`; the per-seed
  `Instruction: Empty` line is the env's own field and is not used. Offline, swapping
  the prompt changed open-loop error by <10 %, since each checkpoint saw one prompt.
- **Cameras**: head always; wrist only for `insert_tube` / `lift_can`. This is
  hard-coded to the training setup, independent of `policy/task_settings.json`.
- **Tactile**: `rgb_marker` of the left and right GelSight Mini pads, as one group
  `[left, right]`. They go to FTP-1's own T3-based tokenizer (in each checkpoint's
  `hpt_tokenizer/`), not to `UniVTAC/encoder/`.
- **State**: absolute `embodiment/joint[:8]` (7 arm + gripper).
- **Actions**: re-infer every step; `chunk[0]` is a placeholder, so `chunk[1:21]` of
  each chunk votes in a temporal ensemble with weights `exp(-0.01·i)`, oldest first.
  The checkpoints are `action_joint_rep: mix` (arm deltas, absolute gripper).

## deploy.yml

| key | default | meaning |
|---|---|---|
| `server_url` | `http://127.0.0.1:5002` | also `FTP1_SERVER` env var |
| `checkpoint_dir` / `domain_name` | `null` | `null`: release checkpoint for the task, domain read from its `normalization/`. Set both to evaluate your own step dir. |
| `instruction` | `null` | `null`: training prompt for the task |
| `use_wrist` | `null` | `null`: wrist only for `insert_tube` / `lift_can` |
| `chunk_first_n` / `ensemble_K` | `20` / `0.01` | temporal ensemble, as `eval_ftp1.sh` |

---

## Known issues

**Scripted pre-move planning failure counts as a policy failure.** Each episode's
`task.reset()` runs `pre_move()` with cuRobo (grasp, lift, approach) before the policy
gets control; those are the `[move:move#N]` lines. When that planning fails
(`Arm motion planning failed ... INVALID_PARTIAL_POSE_COST_METRIC`), `plan_success`
goes `False`, `_step()` becomes a no-op, the policy burns all `step_lim` actions without
the arm moving, the episode is logged `failed` with `actions: 300`, and the video is an
empty ~300-byte file. The official `eval_ftp1.py` counts these as failures too, so keep
them to match its protocol; pass `--expert_check` to skip seeds whose expert fails
(cached in `seeds.json`) when you want a policy-only success rate. Use the same choice
for every policy you compare.

**`Possible version incompatibility ... CudaInterop v0.15 against v0.14`** at Isaac Sim
startup is a carb plugin minor-version notice from the sim's own stack, printed before
the policy is imported. It is unrelated to FTP-1.

**Server terminal looks idle.** Expected: it logs only on `/load`
(`Loading FTP-1 checkpoint ...`, GPU memory, `Loaded: ...`) and errors; per-step
requests are not logged.
