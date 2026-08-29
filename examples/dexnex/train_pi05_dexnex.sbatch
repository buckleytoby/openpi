#!/bin/bash
#SBATCH --job-name=pi05_dexnex_erlenmeyer
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Targets ~12h wall clock on an H100 (measured 1.4s/iter at batch_size=32,
# vs 3.4s/iter on an A100 -- see docs/quest/quest_instructions.md in
# avatar_master's timing smoke tests). --num-train-steps=28000 leaves
# ~30-45min of buffer under the 12h --time limit for: container/
# checkpoint-restore startup (~5min), the one-time XLA compile at step 0
# (~1-2min), and periodic checkpoint saves (~28 saves at default
# --save-interval=1000, ~60s each worst case per the smoke test's
# async-save timings). If your measured rate drifts (e.g. a different
# node/GPU), recompute:
#   steps = (43200 - startup(~300s) - n_saves*60s) / seconds_per_iter
# and adjust --num-train-steps accordingly, or just lower --time margin.
#
# WANDB_API_KEY must already be exported in your shell (or ~/.bashrc /
# a sourced dotfile) before sbatch'ing this -- see
# docs/quest/quest_instructions.md's wandb section (in avatar_master) for
# why NOT to use `wandb login` here (version mismatch with the new
# ~86-char API key format).
#
# SBATCH_ACCOUNT (Slurm allocation/project id, e.g. "p52914") must also be
# exported in your shell/~/.bashrc before sbatch'ing this -- sbatch reads
# SBATCH_<OPTION> env vars as defaults for options not set via an explicit
# #SBATCH directive or CLI flag, which is why there's no
# "#SBATCH --account=..." line above: #SBATCH directives are parsed
# literally (no shell variable expansion), so this is the only way to keep
# the actual account id out of this file (and out of a public repo) while
# still letting sbatch pick it up automatically. Re-used below (as a plain
# shell var) to build the /projects/$SBATCH_ACCOUNT/... paths too, since
# Quest's convention is one /projects/<account> dir per allocation.
for var in WANDB_API_KEY SBATCH_ACCOUNT; do
    if [ -z "${!var}" ]; then
        echo "$var is not set in the submitting shell's environment -- export it (e.g. in ~/.bashrc) before sbatch'ing this script." >&2
        exit 1
    fi
done

module load singularityce/4.3.1-gcc-8.5.0

cd "$(dirname "$0")/../.." || exit 1

singularity exec --nv --overlay /projects/$SBATCH_ACCOUNT/dexnex/overlay.img \
  --bind $PWD:/app \
  --bind /projects/$SBATCH_ACCOUNT/dexnex/task_24_erlenmeyer_flask:/data/my_dataset/dexnex/task_24_erlenmeyer_flask \
  --bind /scratch/$USER/openpi_cache:/openpi_assets \
  --env HF_LEROBOT_HOME=/data/my_dataset \
  --env OPENPI_DATA_HOME=/openpi_assets \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --pwd /app \
  openpi_server.sif \
  uv run scripts/train.py pi05_dexnex_erlenmeyer \
    --exp-name=run1_h100_12h --overwrite \
    --batch-size=32 --num-workers=8 \
    --num-train-steps=28000 --save-interval=1000
