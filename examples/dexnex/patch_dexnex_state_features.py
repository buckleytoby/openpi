"""
Patch an already-converted dexnex LeRobot dataset in place to:
  - append biotac_lh channels [0, 1, 4] (thumb/index/middle -- the other two are dummy
    sensors that are always zero) to `state`, making it 10-dim
  - add a per-episode `env_image_valid` flag (see convert_dexnex_data_to_lerobot.py for why)

This only rewrites the `state` column and adds `env_image_valid`; the image columns are
passed through unchanged (no JPEG2000 decode / PNG re-encode), so this is much faster
than rerunning the full conversion script, which spends nearly all its time on images.

Usage:
HF_LEROBOT_HOME=/media/dexnex_ssd/datasets python3.11 examples/dexnex/patch_dexnex_state_features.py \
    --data_path /media/dexnex_ssd/datasets/task_24_erlenmeyer_flask.zarr \
    --repo_name dexnex/task_24_erlenmeyer_flask
"""

import json

import imagecodecs.numcodecs as _imagecodecs_numcodecs
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro
import zarr

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

_imagecodecs_numcodecs.register_codecs()

# Of the 5 biotac_lh channels, only these carry real signal (thumb, index, middle);
# the rest are dummy sensors that are always zero.
BIOTAC_CHANNELS = [0, 1, 4]


def main(
    data_path: str = "/media/dexnex_ssd/datasets/task_24_erlenmeyer_flask.zarr",
    repo_name: str = "dexnex/task_24_erlenmeyer_flask",
):
    dataset_root = HF_LEROBOT_HOME / repo_name
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    if info["features"]["state"]["shape"][0] == 10 and "env_image_valid" in info["features"]:
        print("Dataset already has the 10-dim state and env_image_valid feature; nothing to do.")
        return

    root = zarr.open(data_path, mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], episode_ends[:-1]])

    parquet_files = sorted(dataset_root.glob("data/chunk-*/*.parquet"))
    for pq_path in tqdm.tqdm(parquet_files, desc="Patching episodes"):
        table = pq.read_table(pq_path)
        episode_index = table["episode_index"][0].as_py()

        # "index" is the global row index across the whole dataset, assigned in the same
        # order frames were added during conversion -- which matches the zarr row order.
        global_idx = np.asarray(table["index"].to_pylist())

        biotac = np.asarray(data["biotac_lh"].oindex[global_idx])[:, BIOTAC_CHANNELS].astype(np.float32)

        old_state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        new_state = np.concatenate([old_state, biotac], axis=-1)

        ep_s, ep_e = int(starts[episode_index]), int(episode_ends[episode_index])
        env_image_valid = int(np.any(data["env_image"][ep_s:ep_e]))
        env_image_valid_col = np.full((len(table),), env_image_valid, dtype=np.int64)

        new_state_array = pa.FixedSizeListArray.from_arrays(
            new_state.reshape(-1).astype(np.float32), new_state.shape[-1]
        )
        env_valid_array = pa.FixedSizeListArray.from_arrays(env_image_valid_col, 1)

        table = table.set_column(table.schema.get_field_index("state"), "state", new_state_array)
        if "env_image_valid" in table.column_names:
            table = table.set_column(
                table.schema.get_field_index("env_image_valid"), "env_image_valid", env_valid_array
            )
        else:
            table = table.append_column("env_image_valid", env_valid_array)

        pq.write_table(table, pq_path)

    info["features"]["state"]["shape"] = [10]
    info["features"]["env_image_valid"] = {"dtype": "int64", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info, indent=4))

    print(
        "Done. Note: meta/episodes_stats.jsonl was not regenerated -- openpi's "
        "compute_norm_stats.py recomputes normalization stats itself and does not "
        "rely on this file. If you use other lerobot tooling that reads cached "
        "per-episode stats, regenerate episodes_stats.jsonl separately."
    )


if __name__ == "__main__":
    tyro.cli(main)
