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

## Data & normalization

- Suites: `libero_{10,goal,object,spatial}_no_noops` in LeRobot format
  (mixed together for a single model).
- Normalization is **mean/std** (FastWAM recipe), per suite — *not* the
  q01/q99 min-max used by the original `data/libero_dataset.py`. Deployment
  must de-normalize with the same stats (`configs/ltx_model/libero/libero_fastwam_mix.json`).
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

## Training

Two stages (LTX): **video adaptation** (`train_mode: video_only`) →
**action post-training** (`action_full`). The action expert is warm-started by
per-dimension linear interpolation (`scripts/interp_init_action_expert.py`).

```bash
# launches torchrun over all visible GPUs; DeepSpeed ZeRO-2 + bf16
bash scripts/train.sh main.py configs/ltx_model/libero/action_model_libero_fastwam_mix.yaml

# Cosmos video-adaptation variant
bash scripts/train.sh main.py configs/cosmos_model/libero/video_model_libero_cosmos.yaml
```

Cosmos notes: `in_channels 17` (16 Wan latent + 1 condition mask; +1 padding
→ 18 into `patch_embed`), `chunk` must be `4n+1` (Wan VAE is 4× temporal),
`pixel_wise_timestep: False` (Cosmos reshapes the timestep per frame). Only
`patch_embed.proj.weight` is re-initialized when loading the GE-Sim base.

## Evaluation

Standard LIBERO closed-loop (50 trials/task in the paper protocol):

```bash
python experiments/eval_libero.py \
  --config_file configs/ltx_model/libero/action_model_libero_fastwam_eval.yaml \
  --ckpt_path  /path/to/step_50000/diffusion_pytorch_model.safetensors \
  --task_suite_name libero_goal --exec_step 8 --num_trails_per_task 50 --device 0
```

LIBERO-plus robustness (10,030 perturbed tasks, 7 dimensions; sharded &
globally resumable — safe to change worker count mid-run):

```bash
python experiments/eval_libero_plus.py \
  --config_file configs/ltx_model/libero/action_model_libero_fastwam_eval.yaml \
  --ckpt_path  /path/to/step_50000/diffusion_pytorch_model.safetensors \
  --out_dir    /path/to/results --shard 0 --num_shards 6 --device 0
```

Both need `MUJOCO_GL=egl` and LIBERO / LIBERO-plus on `PYTHONPATH`.

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
