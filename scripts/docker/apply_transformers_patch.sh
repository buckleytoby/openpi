#!/bin/bash
# Applies the src/openpi/models_pytorch/transformers_replace patch to the installed
# `transformers` package inside the container's venv. The Docker image bakes this in at
# build time; the Singularity/Apptainer .def (serve_policy.def) does not (see the note in
# that file), so run this once per container invocation before any script that imports
# `transformers`, e.g.:
#
#   singularity exec --nv --bind $PWD:/app --pwd /app openpi_server.sif \
#     bash -c "scripts/docker/apply_transformers_patch.sh && uv run scripts/train.py ..."
set -euo pipefail

TRANSFORMERS_DIR="$(/.venv/bin/python -c 'import transformers; print(transformers.__file__)' | xargs dirname)"
cp -r /app/src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"
