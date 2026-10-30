# Training run: `pi05_dexnex_erlenmeyer_cotrain_balanced_rot_costbal_views_layout`

Handoff for running this fine-tune on dexbeast (written 2026-10-30 from the lab PC "alien"). The data is ready and
audited. The training run is the only step left. The first attempt ran out of GPU memory and needs settings fixed.

## What this run is

- **Model:** pi0.5, PyTorch, fine-tuned from the pi0.5 base weights:
  `/home/dexbeast/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_dexnex`. The 3 earlier runs used the same
  path; their wandb configs record it.
- **Config:** `pi05_dexnex_erlenmeyer_cotrain_balanced_rot_costbal_views_layout`, defined in
  `src/openpi/training/config.py`. It's the costbal+DR co-train config with two changes:
  1. **New dataset** `dexnex/task_24_erlenmeyer_flask_cotrain_layout` (in `~/datasets`). The real data is the same
     1,249 episodes. The sim episodes were re-rendered with the table and tray moved relative to the robot at random per
     episode (±5 cm and ±5° for the table, ±2 cm and ±3° for the tray on it), plus domain-randomized visuals. The sim
     zarrs are in `~/datasets/layout/`: 751 `aiet_grip` and 5,429 `aiet_grip_lift` episodes.
  2. **View-weighted sampling** (`DataConfig.view_weights`, in `src/openpi/training/data_loader.py`). Two per-frame
     views of sim frames only (`meta/frame_views.npz`) get extra batch share:
     - `flush_corner` (15%): episodes whose pellet started pressed against the tray (flush, corner or wall starts);
     - `fine_align` (20%): a fingertip within 2 cm of the pellet, up to the lift.

     The other 65% keeps the costbal mix (source shares real 0.5 / `aiet_grip` 0.25 / `aiet_grip_lift` 0.25, sim
     sources cost-balanced). The audit's measured shares per batch: real 32.5%, `aiet_grip` 19.6%, `aiet_grip_lift`
     47.9%; frames in `flush_corner` 31.2%, in `fine_align` 38.1%. The user accepted this mix; don't change it.
- **Norm stats** for this config are already computed: `assets/<config>/dexnex/task_24_erlenmeyer_flask_cotrain_layout/norm_stats.json`.

## Already done (don't redo)

1. The dataset was converted (`convert_dexnex_data_to_lerobot.py`).
2. `meta/frame_start_cost.npy` was written (`write_start_costs.py`) and `meta/frame_views.npz` (`write_frame_views.py`).
3. Norm stats were computed (`scripts/compute_norm_stats.py`).
4. The audit passed: `python -m examples.dexnex.audit_cotrain_views`, 19/21 checks. The 2 "FAIL"s are false alarms.
   They compare domain-randomization colour spread against `~/datasets/aiet_grip*_sim_to_real.zarr`, which were
   already domain-randomized, so the spread doesn't increase. DR is present (spread ~30/255).

## The problem to fix: out of GPU memory at batch size 128

The first attempt loaded the weights correctly ("Loaded PyTorch weights from …/pi05_base_pytorch_dexnex"), then hit
`torch.OutOfMemoryError` on the first step. That's batch size 128 (the config's default) on a 31 GB GPU, with
gradient checkpointing on and bfloat16. The process itself held about 30 GB; nothing else was using the GPU. The
earlier runs on this machine with the same base config must have used different settings. **Copy them, so this run is
comparable to the costbal+DR run** (`pi05_dexnex_erlenmeyer_cotrain_balanced_rot_costbal/balanced_rot_costbal_dr_local1`).
Their wandb configs, in `~/libraries/openpi/wandb/run-*/files/config.yaml`, have the values, e.g.:

```
cd ~/libraries/openpi
for k in batch_size num_train_steps pytorch_training_precision fsdp_devices; do echo "== $k"; grep -h -A1 "^$k:" wandb/run-*/files/config.yaml | grep value | sort | uniq -c; done
```

Use exactly the earlier runs' batch size (and any other non-default values they set). The LR schedule (warmup 1000,
peak 5e-5, 30k steps) was tuned at 128, so if the batch size differs from 128, keep whatever schedule the earlier runs
used. Check the GPU is free first (`nvidia-smi`). If memory is still tight at the same batch size,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps: the OOM showed 4.8 GB reserved but unallocated.

## Command template

Run from `~/libraries/openpi`, detached so the SSH session can close:

```
WEIGHTS=/home/dexbeast/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_dexnex
HF_LEROBOT_HOME=~/datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup uv run scripts/train_pytorch.py \
    pi05_dexnex_erlenmeyer_cotrain_balanced_rot_costbal_views_layout \
    --exp_name views_layout_local1 --pytorch_weight_path "$WEIGHTS" --overwrite --batch_size <the earlier runs' value> \
    > ~/train_views_layout.log 2>&1 &
echo $! > ~/train_views_layout.pid
```

- **Single GPU: `uv run`, not `torchrun`.** The source/view-weighted sampler raises `NotImplementedError` under DDP.
- **`--overwrite`:** the failed attempts already created `checkpoints/<config>/views_layout_local1`. Use `--resume`
  instead of `--overwrite` only to continue a run that has saved checkpoints.
- **Always pass `--pytorch_weight_path`.** Without it, `train_pytorch.py` ignores the config's JAX weight loader and
  starts from random weights.

## How to confirm it's right

In `~/train_views_layout.log`:
- `Loaded PyTorch weights from /home/dexbeast/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_dexnex`;
- the sampler's lines: each source's share, `balanced over start-cost bins`, and `view 'flush_corner'` / `view
  'fine_align'` with their frame counts and "+15% / +20% of each batch";
- the data config showing `repo_id='dexnex/task_24_erlenmeyer_flask_cotrain_layout'`,
  `view_weights={'flush_corner': 0.15, 'fine_align': 0.2}`, and `angle_offset=-1.5707963267948966` (the `_rot`
  configs);
- training steps advancing with a falling loss.

## Background

- Why: the real-robot policy is still bad at flush and corner pellet placements and at the fine alignment before the
  grasp. And the policy stopped short of the tray when the real shop stand was closer to the robot than usual, so the sim
  data now varies the table and tray placement relative to the robot.
- The sim views were made on alien by `avatar_intelligence/.../tree_exploration/sim_views.py`. The re-render with
  random layouts was done by `export_layout_batches.sh`. Nothing on dexbeast needs those.
- Checkpoints go to `checkpoints/pi05_dexnex_erlenmeyer_cotrain_balanced_rot_costbal_views_layout/views_layout_local1/<step>/`.
  The user copies the ones to evaluate to alien and loads them in the openpi inference GUI
  (`avatar_intelligence/inference/openpi_models.json`).
