#!/bin/bash
#SBATCH --job-name=pi05_dexnex_erlenmeyer_cotrain
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
# 128G, not 64G: a run at 64G got oom_kill'd by Slurm right at the first
# checkpoint save (step 1000) -- orbax's async checkpoint save transfers the
# full training state (params+optimizer, ~25GB combined) to HOST memory,
# which stacks on top of whatever the 8 dataloader workers + JAX/XLA host
# buffers were already using, and that in-flight async-save buffer isn't
# necessarily freed before the next step's work continues. This is host
# (CPU) RAM getting oom_kill'd by Slurm's cgroup, a completely different
# failure mode from the earlier GPU_0_bfc RESOURCE_EXHAUSTED (VRAM) OOM.
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Sim/real co-training (pi05_dexnex_erlenmeyer_cotrain): train_pi05_dexnex.sbatch with the combined
# dataset (real demos + SuperDex sim successes: convert_dexnex_data_to_lerobot.py --sim-paths) and its
# config. Same budget, environment variables and memory notes as there (read its comments).
#
# Before submitting, on Quest:
# - the combined LeRobot dataset at /projects/$SBATCH_ACCOUNT/dexnex/task_24_erlenmeyer_flask_cotrain
#   (rsync from omnid's $HF_LEROBOT_HOME/dexnex/task_24_erlenmeyer_flask_cotrain);
# - its norm stats at $OPENPI_REPO_DIR/assets/pi05_dexnex_erlenmeyer_cotrain/dexnex/task_24_erlenmeyer_flask_cotrain/
#   norm_stats.json (assets/ is gitignored: rsync it too, after scripts/compute_norm_stats.py locally).
for var in WANDB_API_KEY SBATCH_ACCOUNT OPENPI_REPO_DIR; do
    if [ -z "${!var}" ]; then
        echo "$var is not set in the submitting shell's environment -- export it (e.g. in ~/.bashrc) before sbatch'ing this script." >&2
        exit 1
    fi
done

module load singularityce/4.3.1-gcc-8.5.0

cd "$OPENPI_REPO_DIR" || exit 1

# Continuing a finished run on a grown dataset: the same --exp-name with --resume picks up its latest checkpoint (weights,
# optimizer, EMA; the data loader isn't restored -- the current dataset is used) and trains to NUM_TRAIN_STEPS (total,
# not additional): e.g. NUM_TRAIN_STEPS=55998 sbatch ... runs 27999 -> 55997 (another 27,999 steps, ~12 h on an H100).
# pi05_dexnex_erlenmeyer_cotrain's learning rate is constant after warmup (peak = decay = 5e-5), so going past its
# 30k decay_steps changes nothing.

# LeRobot's load_dataset("parquet") first writes an Arrow copy of the whole dataset (~34 GB) into the HF datasets
# cache -- by default ~/.cache/huggingface/datasets, i.e. the Quest home directory, whose quota it overflows
# ("OSError: [Errno 122] Disk quota exceeded", job 4350183). Keep that cache on scratch instead.
mkdir -p /scratch/$USER/hf_datasets_cache

singularity exec --nv --overlay /projects/$SBATCH_ACCOUNT/dexnex/overlay.img \
  --bind $PWD:/app \
  --bind /projects/$SBATCH_ACCOUNT/dexnex/task_24_erlenmeyer_flask_cotrain:/data/my_dataset/dexnex/task_24_erlenmeyer_flask_cotrain \
  --bind /scratch/$USER/openpi_cache:/openpi_assets \
  --bind /scratch/$USER/hf_datasets_cache:/hf_datasets_cache \
  --env HF_LEROBOT_HOME=/data/my_dataset \
  --env HF_DATASETS_CACHE=/hf_datasets_cache \
  --env OPENPI_DATA_HOME=/openpi_assets \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --pwd /app \
  openpi_server.sif \
  uv run scripts/train.py pi05_dexnex_erlenmeyer_cotrain \
    --exp-name=cotrain1_h100_12h --resume \
    --batch-size=32 --num-workers=8 \
    --num-train-steps=${NUM_TRAIN_STEPS:-28000} --save-interval=1000
