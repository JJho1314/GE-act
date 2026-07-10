# GE-Act on LIBERO — Training & Evaluation

This fork adds a full **LIBERO** training + closed-loop evaluation pipeline for
GE-Act, with two interchangeable video backbones:

- **LTX-Video 2B** (the default GE-Act backbone), and
- **Cosmos / GE-Sim** (`MultiViewCosmosTransformer3DModel` + Wan VAE + T5-1024).

It also ships a **LIBERO-plus** robustness evaluator and several correctness
fixes. Everything is driven by the existing generic trainer
(`runner/ge_trainer.py`) — backbones are swapped purely through config.

---

## What's here

| Area | File(s) |
|---|---|
| Dataset (LeRobot / mp4, PyAV decode) | `data/lerobot_like_dataset.py` |
| Action-expert interpolation init (FastWAM recipe) | `scripts/interp_init_action_expert.py` |
| Closed-loop eval (standard LIBERO) | `experiments/eval_libero.py` |
| Robustness eval (LIBERO-plus) | `experiments/eval_libero_plus.py` |
| LTX training / eval configs | `configs/ltx_model/libero/` |
| Cosmos video-adaptation config | `configs/cosmos_model/libero/` |

## Environment

- **Training** — the GE base env (torch 2.7 + cu126, diffusers 0.32,
  deepspeed 0.15, `ftfy`, `easydict`). Multi-GPU via `torchrun` + DeepSpeed.
- **Evaluation** — a separate env with the LIBERO simulator stack
  (`robosuite`, `bddl`, `egl_probe`, EGL offscreen rendering). Closed-loop
  eval is **CPU/sim-bound**, so GPU utilization per process is low; pack
  several eval processes per GPU to fill it.
  - LIBERO is an implicit namespace package (no `__init__.py`); put it on
    `PYTHONPATH` rather than `pip -e`.
  - Always set `MUJOCO_GL=egl` (and `EGL_DEVICE_ID` to the target GPU).

## Data & normalization

- Suites: `libero_{10,goal,object,spatial}_no_noops` in LeRobot format
  (mixed together for a single model).
- Normalization is **mean/std** (FastWAM recipe), per suite — *not* the
  q01/q99 min-max used by the original `data/libero_dataset.py`. Deployment
  must de-normalize with the same stats
  (`configs/ltx_model/libero/libero_fastwam_mix.json`).
- Action layout is `[7 action ; 8 state] = 15` dims; the first 7 are the action.

## Action space (important)

For LIBERO the dataset action's position/rotation are already **deltas in the
controller `[-1, 1]` input space** — feed them straight to `env.step`, no
absolute→delta conversion. The **only** transform needed is the gripper:

```
data gripper ∈ [0, 1]  (0 = close, 1 = open)
env  gripper ∈ [-1, 1] (-1 = open,  +1 = close)   →   g_env = 1 - 2 * g
```

Missing this makes the gripper stay closed and tanks the success rate; adding
it takes `libero_goal` from ~30% to 100%.

---

# Workflow A — LIBERO training

GE-Act trains in **two stages**: adapt the video tower to the LIBERO domain,
then post-train the action expert. All launches use:

```bash
# torchrun over all visible GPUs; DeepSpeed ZeRO-2 + bf16
bash scripts/train.sh main.py <config.yaml>
```

Global batch = `batch_size (per GPU) × #GPUs × gradient_accumulation_steps`.
Our runs used **128** (e.g. 16 × 8 × 1 for the action stage, or 8 × 8 × 2 for
Cosmos video adaptation). Checkpoints land in
`<output_dir>/<timestamp>/step_<N>/` as diffusers format
(`config.json` + `diffusion_pytorch_model.safetensors`), every
`steps_to_save` steps.

### Stage 1 — video adaptation (`video_only`)

Adapts the pretrained backbone to LIBERO video. Starts from the GE base
checkpoint (`GE_base_fast_v0.1.safetensors` for LTX).

```bash
bash scripts/train.sh main.py \
  configs/ltx_model/libero/video_model_libero_fastwam_mix.yaml
```

Run to ~20–30k steps (loss ≈ 0.02, open-loop videos track the ground truth).
Pick a checkpoint, e.g. `step_20000`, for the next step.

### Stage 2a — action-expert interpolation init

Warm-start the action expert from the stage-1 video weights by per-dimension
linear interpolation (FastWAM recipe: `align_corners=True`, scale
`α = √(dv/da)`, skip `action_proj_in/out`):

```bash
python scripts/interp_init_action_expert.py \
  --src  <stage1>/step_20000/diffusion_pytorch_model.safetensors \
  --dst  GE_vadapt20k_actinterp.safetensors
```

### Stage 2b — action post-training (`action_full`)

Point the config's `diffusion_model.model_path` at the interp-init weight and
train the full model (video tower + action expert):

```bash
bash scripts/train.sh main.py \
  configs/ltx_model/libero/action_model_libero_fastwam_mix.yaml
```

Run to ~50k steps → the final GE-Act policy used for evaluation.

### Cosmos backbone variant

Same Stage-1 recipe, different config (Cosmos = Wan VAE + T5-1024 +
`MultiViewCosmosTransformer3DModel`, base `ge_sim_cosmos_v0.1.safetensors`):

```bash
bash scripts/train.sh main.py \
  configs/cosmos_model/libero/video_model_libero_cosmos.yaml
```

Cosmos notes: `in_channels 17` (16 Wan latent + 1 condition mask; `+1` padding
→ 18 into `patch_embed`); `chunk` must be `4n+1` (Wan VAE is 4× temporal);
`pixel_wise_timestep: False` (timestep is reshaped per frame). Only
`patch_embed.proj.weight` is re-initialized when loading the GE-Sim base.
To resume/continue, set `diffusion_model.model_path` to a saved
`step_<N>/diffusion_pytorch_model.safetensors` (loads weights only — the step
counter and optimizer state reset, LR re-warms).

---

# Workflow B — LIBERO closed-loop inference / evaluation

Runs the trained policy in the LIBERO simulator and reports success rate. Uses
the eval config (`action_model_libero_fastwam_eval.yaml`), which points at the
local backbone components and the mixed-suite stats.

```bash
export MUJOCO_GL=egl EGL_DEVICE_ID=0
export PYTHONPATH=.:/path/to/LIBERO

python experiments/eval_libero.py \
  --config_file configs/ltx_model/libero/action_model_libero_fastwam_eval.yaml \
  --ckpt_path   /path/to/step_50000/diffusion_pytorch_model.safetensors \
  --output_dir  /path/to/results/libero_goal \
  --task_suite_name libero_goal \
  --exec_step 8 --num_trails_per_task 50 --threshold 20 --device 0
```

Per-suite settings (paper protocol = 50 trials/task):

| suite | `--threshold` | max steps |
|---|---|---|
| libero_spatial | 30 | 220 |
| libero_object  | 30 | 280 |
| libero_goal    | 20 | 300 |
| libero_10      | 20 | 520 |

- `--exec_step 8`: 8 env steps are executed per policy call (the video tower
  runs once, then 10-step action denoising produces the chunk).
- Output: per-suite `inference_*.txt` (final total success rate) + rollout mp4s.
- Run the four suites in parallel across GPUs; each process is sim-bound, so
  several fit on one GPU.

---

# Workflow C — LIBERO-plus robustness evaluation (inference only)

**LIBERO-plus is an evaluation-only benchmark — you do not train on it.** It is
a drop-in `libero` replacement with **10,030 perturbed tasks** across 7
dimensions (Objects Layout / Camera Viewpoints / Robot Initial States /
Language Instructions / Light Conditions / Background Textures / Sensor Noise),
1 trial/task. You run the **same LIBERO-trained checkpoint** through it.

### 1. Point at LIBERO-plus (non-interactive)

LIBERO-plus *is* a real package (has `__init__.py`) and must win over any
standard LIBERO on `sys.path` — `experiments/eval_libero_plus.py` inserts it
first via `LIBERO_PLUS_ROOT`. Its `__init__` otherwise prompts interactively,
so pre-create `~/.libero_plus/config.yaml`:

```yaml
benchmark_root: /path/to/LIBERO-plus/libero/libero
bddl_files:     /path/to/LIBERO-plus/libero/libero/bddl_files
init_states:    /path/to/LIBERO-plus/libero/libero/init_files
datasets:       /path/to/LIBERO-plus/libero/libero/../datasets
assets:         /path/to/LIBERO-plus/libero/libero/assets
```

```bash
export MUJOCO_GL=egl
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus
export LIBERO_CONFIG_PATH=$HOME/.libero_plus
# optional: pip install wand scikit-image  (wand only affects the motion-blur subset)
```

### 2. Launch sharded across GPUs

Work is the `(suite, task_id)` list; each shard takes `idx % num_shards`. Pack
multiple shards per GPU to fill it (sim-bound; ~15 GB each):

```bash
# e.g. 6 shards over 2 GPUs — GPU0: shards 0,1,2 ; GPU1: shards 3,4,5
python experiments/eval_libero_plus.py \
  --config_file configs/ltx_model/libero/action_model_libero_fastwam_eval.yaml \
  --ckpt_path   /path/to/step_50000/diffusion_pytorch_model.safetensors \
  --out_dir     /path/to/results/libero_plus \
  --shard 0 --num_shards 6 --device 0
```

### 3. Resume / rescale freely

Each shard writes `results_shard<N>.json` incrementally. Resume is **global**:
on start it reads *all* `results_shard*.json` and skips any `(suite, task_id)`
already done — so you can crash, change the worker count, or re-balance GPUs
and never repeat a rollout. Just relaunch the shards you want.

### 4. Aggregate

Merge every `results_shard*.json` `by_task` record, accumulating by `suite`
and by `category` (the 7 dimensions):

```python
import json, glob, collections
suite = collections.Counter(); suite_n = collections.Counter()
cat   = collections.Counter(); cat_n   = collections.Counter()
for pf in glob.glob("results/libero_plus/results_shard*.json"):
    for r in json.load(open(pf))["by_task"]:
        suite[r["suite"]] += r["successes"]; suite_n[r["suite"]] += 1
        cat[r["category"]] += r["successes"]; cat_n[r["category"]] += 1
for k in cat:   print(f"{k:24s} {cat[k]/cat_n[k]:.3f}")
```

---

## Results (our 50k GE-Act, LTX backbone)

**Standard LIBERO** — 50 trials/task, avg **0.981**:

| spatial | object | libero_10 | goal |
|---|---|---|---|
| 0.996 | 0.992 | 0.976 | 0.960 |

**LIBERO-plus** — 1 trial/task over all 10,030, overall **0.782**:

| dimension | rate | | dimension | rate |
|---|---|---|---|---|
| Light Conditions | 0.957 | | Objects Layout | 0.824 |
| Background Textures | 0.863 | | Language Instructions | 0.800 |
| Sensor Noise | 0.825 | | Robot Initial States | 0.781 |
| | | | **Camera Viewpoints** | **0.505** |

Takeaway: robust to appearance perturbations (light/texture), fragile to
geometric ones — camera-viewpoint shifts are by far the weakest axis,
consistent with the LIBERO-plus paper's finding that VLAs lean heavily on a
fixed camera geometry.

## Fixes included

- `data/lerobot_like_dataset.py`: inference-branch index concat used numpy
  element-wise add instead of list concat (`.tolist()`), causing a
  `(4,)(9,)` broadcast error in open-loop / closed-loop inference.
- `experiments/eval_libero.py`: align eval with FastWAM training — per-suite
  stat selection, mean/std (de)norm, and the gripper transform above.
- `runner/ge_trainer.py`: guard `validate()`/checkpoint-save with
  `global_step > 0`. At `global_step == 0`, `global_step % steps_to_val == 0`
  is always true; with `gradient_accumulation_steps > 1` the first optimizer
  step leaves `global_step` at 0 and this fired `validate()` at step 0
  (a crash for the Cosmos path). `grad_accum == 1` dodged it by chance.
