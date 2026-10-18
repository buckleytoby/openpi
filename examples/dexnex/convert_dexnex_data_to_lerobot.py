"""
Convert a dexnex zarr dataset (e.g. task_24_erlenmeyer_flask.zarr) to LeRobot format
for openpi fine-tuning.

Assumptions about the zarr layout (verify these against your data before training):
- `data/env_image`, `data/wrist_image`, `data/overhead_image`: uint8 arrays of shape
  (N, 224, 224, 3), one row per timestep, aligned 1:1 with the proprio/action arrays
  below (N must match across all of them -- if the image-writing script referenced when
  this script was written hasn't finished, N for the images will be smaller than N for
  the proprio arrays and this script will raise an assertion error).
- `data/palm_pose_xyz_rpy_states` (N, 6) + `data/gripper_value_states` (N,) + `data/biotac_lh`
  (N, 5) channels [0, 1, 4] (thumb/index/middle -- channels 2 and 3 are dummy sensors,
  always zero): used as the 10-dim `state` (measured/observed).
- `data/palm_pose_xyz_rpy_commands` (N, 6) + `data/gripper_value_commands` (N,): used as
  the 7-dim `actions` (commanded).
- `meta/episode_ends`: 1D int array of cumulative end-indices (exclusive) into the N-length
  timeline above, one entry per episode.

Usage:
HF_LEROBOT_HOME=/media/dexnex_ssd/datasets uv run examples/dexnex/convert_dexnex_data_to_lerobot.py \
    --data_path /media/dexnex_ssd/datasets/task_24_erlenmeyer_flask.zarr \
    --task "pick up the pellet from the tray and place it in the erlenmeyer flask" \
    --fps 10

Sim/real co-training (pi05_dexnex_erlenmeyer_cotrain): add the sim zarrs (same layout -- avatar_intelligence's
tree_exploration/sim_to_real.py writes them, BioTac all zero) and the combined dataset's name:
HF_LEROBOT_HOME=/media/dexnex_ssd/datasets uv run examples/dexnex/convert_dexnex_data_to_lerobot.py \
    --repo-name dexnex/task_24_erlenmeyer_flask_cotrain \
    --sim-paths /media/dexnex_ssd/datasets/aiet_grip_sim_to_real.zarr
Each sim episode is written --sim-repeat times (1: the natural share); every frame carries `is_sim` (0 real, 1 sim:
not a model input, for analysis / per-domain stats). Same prompt for both (no domain marker).

If you want to push your dataset to the Hugging Face Hub, add --push_to_hub.

The resulting dataset is saved to $HF_LEROBOT_HOME / <repo_name> (defaults to
~/.cache/huggingface/lerobot / <repo_name> if HF_LEROBOT_HOME is not set -- set it to
point at the SSD as shown above, since the converted dataset can be large).
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import imagecodecs.numcodecs as _imagecodecs_numcodecs
import numpy as np
import tqdm
import tyro
import zarr

# The env_image/wrist_image/overhead_image arrays are stored with a JPEG2000 codec from
# `imagecodecs`, which doesn't auto-register with numcodecs on import.
_imagecodecs_numcodecs.register_codecs()

REPO_NAME = "dexnex/task_24_erlenmeyer_flask"  # Name of the output dataset, also used for the Hugging Face Hub

# Of the 5 biotac_lh channels, only these carry real signal (thumb, index, middle);
# the rest are dummy sensors that are always zero.
BIOTAC_CHANNELS = [0, 1, 4]


def main(
    data_path: str = "/media/dexnex_ssd/datasets/task_24_erlenmeyer_flask.zarr",
    task: str = "pick up the pellet from the tray and place it in the erlenmeyer flask",
    *,
    fps: int = 10,
    repo_name: str = REPO_NAME,
    sim_paths: list[str] | None = None,
    sim_repeat: int = 1,
    push_to_hub: bool = False,
):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="gofa1_shadow_hand",
        fps=fps,
        features={
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "env_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "overhead_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            # env_image was only added partway through data collection (later ROS bags).
            # Episodes recorded before that have an all-zero env_image; this flag lets the
            # training pipeline mask env_image out for those episodes instead of treating
            # the zeros as real visual input.
            "env_cam_valid": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            # 0: a real demo, 1: a sim episode (co-training; not a model input)
            "is_sim": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    sources = [(data_path, 0, 1)] + [(path, 1, sim_repeat) for path in (sim_paths or [])]
    totals = {0: 0, 1: 0}
    for path, is_sim, repeat in sources:
        for _ in range(repeat):
            totals[is_sim] += _write_zarr(dataset, path, task, is_sim)
    if sim_paths:
        print(f"frames: {totals[0]} real, {totals[1]} sim ({100 * totals[1] / max(totals[0] + totals[1], 1):.1f}% sim)")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["dexnex", "gofa1", "shadow_hand"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


def _open_zarr(path: str):
    """The zarr's data group and episode ends, checked (every array as long as the timeline, episode_ends matching)."""
    root = zarr.open(path, mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta/episode_ends"])

    num_steps = data["palm_pose_xyz_rpy_states"].shape[0]
    for key in (
        "env_image",
        "wrist_image",
        "overhead_image",
        "palm_pose_xyz_rpy_commands",
        "gripper_value_states",
        "gripper_value_commands",
        "biotac_lh",
    ):
        assert data[key].shape[0] == num_steps, (
            f"{path}: expected '{key}' to have {num_steps} rows (matching palm_pose_xyz_rpy_states), "
            f"got {data[key].shape[0]}. If images are still being written by another script, wait for it to finish."
        )
    assert episode_ends[-1] == num_steps, (
        f"{path}: meta/episode_ends last entry ({episode_ends[-1]}) does not match the number of timesteps ({num_steps})."
    )
    return data, episode_ends


def _write_zarr(dataset, path: str, task: str, is_sim: int) -> int:
    """Every episode of the zarr at ``path`` into ``dataset`` (one LeRobot episode each); returns its frame count."""
    data, episode_ends = _open_zarr(path)
    num_steps = int(episode_ends[-1])
    start = 0
    with tqdm.tqdm(total=num_steps, desc=f"Converting {'sim' if is_sim else 'real'} frames", unit="frame") as pbar:
        for episode_idx, end in enumerate(episode_ends):
            end = int(end)
            palm_states = data["palm_pose_xyz_rpy_states"][start:end]
            gripper_states = data["gripper_value_states"][start:end]
            biotac = data["biotac_lh"][start:end][:, BIOTAC_CHANNELS]
            palm_commands = data["palm_pose_xyz_rpy_commands"][start:end]
            gripper_commands = data["gripper_value_commands"][start:end]
            wrist_images = data["wrist_image"][start:end]
            env_images = data["env_image"][start:end]
            overhead_images = data["overhead_image"][start:end]

            state = np.concatenate([palm_states, gripper_states[:, None], biotac], axis=-1).astype(np.float32)
            actions = np.concatenate([palm_commands, gripper_commands[:, None]], axis=-1).astype(np.float32)

            # env_image was not recorded for all episodes (see feature comment above) --
            # treat an episode as invalid for env_image if every pixel in the episode is zero.
            env_cam_valid = np.array([0 if not np.any(env_images) else 1], dtype=np.int64)

            pbar.set_postfix(episode=f"{episode_idx + 1}/{len(episode_ends)}")
            for i in range(end - start):
                dataset.add_frame(
                    {
                        "wrist_image": wrist_images[i],
                        "env_image": env_images[i],
                        "overhead_image": overhead_images[i],
                        "state": state[i],
                        "actions": actions[i],
                        "env_cam_valid": env_cam_valid,
                        "is_sim": np.array([is_sim], dtype=np.int64),
                        "task": task,
                    }
                )
                pbar.update(1)
            dataset.save_episode()
            # LeRobot keeps every saved episode, images embedded, in dataset.hf_dataset (save_episode's
            # concatenate_datasets): ~38 GB of RAM by 230k frames (the real + sim dataset got OOM-killed). Each
            # episode is already on disk as its parquet, and nothing here reads it back: start it empty again.
            dataset.hf_dataset = dataset.create_hf_dataset()
            start = end
    return num_steps


if __name__ == "__main__":
    tyro.cli(main)
