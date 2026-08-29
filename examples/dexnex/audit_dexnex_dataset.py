"""
Audit a converted dexnex LeRobot dataset against its source zarr dataset.

Checks:
- Episode/frame counts match between LeRobot meta and the source zarr.
- Per-episode lengths match.
- Task string is as expected.
- Full-dataset scan of `state`/`actions` for NaN/Inf values.
- Per-episode check (from episodes_stats.jsonl) for any image stream that is entirely
  zero for an episode (e.g. a camera that wasn't recording during that part of collection).
- Random spot-checks: for a sample of frames, re-read the LeRobot parquet row and compare
  its `state`/`actions` values and decoded images against the corresponding row in the
  source zarr.

Usage:
HF_LEROBOT_HOME=/media/dexnex_ssd/datasets python3.11 examples/dexnex/audit_dexnex_dataset.py \
    --data_path /media/dexnex_ssd/datasets/task_24_erlenmeyer_flask.zarr \
    --repo_name dexnex/task_24_erlenmeyer_flask
"""

import io
import json
import pathlib

import imagecodecs.numcodecs as _imagecodecs_numcodecs
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro
import zarr
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

_imagecodecs_numcodecs.register_codecs()

IMAGE_KEYS = ("wrist_image", "env_image", "overhead_image")
BIOTAC_CHANNELS = [0, 1, 4]


def _episode_spans(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return list(zip(starts.tolist(), episode_ends.tolist(), strict=True))


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stat_is_all_zero(stat: dict) -> bool:
    flat = np.asarray(stat["mean"]).ravel()
    return bool(np.all(flat == 0.0)) and bool(np.all(np.asarray(stat["max"]).ravel() == 0.0))


def main(
    data_path: str = "/media/dexnex_ssd/datasets/task_24_erlenmeyer_flask.zarr",
    repo_name: str = "dexnex/task_24_erlenmeyer_flask",
    task: str = "pick up the pellet from the tray and place it in the erlenmeyer flask",
    num_spot_checks: int = 30,
    image_diff_threshold: float = 5.0,
    seed: int = 0,
):
    dataset_root = HF_LEROBOT_HOME / repo_name
    rng = np.random.default_rng(seed)
    errors: list[str] = []
    warnings: list[str] = []

    # --- Load LeRobot metadata ---
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    episodes_stats = _read_jsonl(dataset_root / "meta" / "episodes_stats.jsonl")

    # --- Load source zarr ---
    root = zarr.open(data_path, mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta/episode_ends"])
    num_steps = data["palm_pose_xyz_rpy_states"].shape[0]
    spans = _episode_spans(episode_ends)

    # 1. Episode / frame counts
    if info["total_episodes"] != len(spans):
        errors.append(f"total_episodes mismatch: info.json={info['total_episodes']} zarr episodes={len(spans)}")
    if info["total_frames"] != num_steps:
        errors.append(f"total_frames mismatch: info.json={info['total_frames']} zarr num_steps={num_steps}")
    if len(episodes) != len(spans):
        errors.append(f"episodes.jsonl has {len(episodes)} entries, expected {len(spans)}")

    # 2. Per-episode length match
    for i, ((s, e), ep) in enumerate(zip(spans, episodes, strict=False)):
        expected_len = e - s
        if ep["length"] != expected_len:
            errors.append(f"episode {i}: length mismatch lerobot={ep['length']} zarr={expected_len}")

    # 3. Task string
    if len(tasks) != 1:
        warnings.append(f"expected exactly 1 task, found {len(tasks)}: {[t['task'] for t in tasks]}")
    elif tasks[0]["task"] != task:
        errors.append(f"task mismatch: expected {task!r}, got {tasks[0]['task']!r}")

    # 4. Per-episode all-zero image check (from precomputed stats -- cheap, no image decode needed)
    zero_image_episodes: dict[str, list[int]] = {k: [] for k in IMAGE_KEYS}
    for ep_stat in episodes_stats:
        ep_idx = ep_stat["episode_index"]
        for key in IMAGE_KEYS:
            if key in ep_stat["stats"] and _stat_is_all_zero(ep_stat["stats"][key]):
                zero_image_episodes[key].append(ep_idx)
    for key, ep_list in zero_image_episodes.items():
        if ep_list:
            warnings.append(
                f"{key}: {len(ep_list)}/{len(spans)} episodes are entirely zero-valued "
                f"(camera likely absent for those episodes) -- e.g. episodes {ep_list[:10]}"
                f"{'...' if len(ep_list) > 10 else ''}"
            )

    # 5. Full-dataset scan for NaN/Inf in state/actions
    nan_count = 0
    total_rows_scanned = 0
    parquet_files = sorted(dataset_root.glob("data/chunk-*/*.parquet"))
    for pq_file in tqdm.tqdm(parquet_files, desc="Scanning state/actions for NaN/Inf"):
        table = pq.read_table(pq_file, columns=["state", "actions"])
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        total_rows_scanned += len(state)
        bad = ~np.isfinite(state).all(axis=-1) | ~np.isfinite(actions).all(axis=-1)
        nan_count += int(bad.sum())
    if nan_count:
        errors.append(f"found {nan_count}/{total_rows_scanned} rows with NaN/Inf in state or actions")

    # 6. Random spot-checks against source zarr
    chunks_size = info["chunks_size"]
    checked = 0
    for _ in tqdm.tqdm(range(num_spot_checks), desc="Spot-checking frames"):
        ep_idx = int(rng.integers(0, len(spans)))
        s, e = spans[ep_idx]
        if e <= s:
            continue
        frame_idx = int(rng.integers(s, e))
        local_idx = frame_idx - s

        chunk_idx = ep_idx // chunks_size
        pq_path = dataset_root / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        table = pq.read_table(pq_path)
        row = table.slice(local_idx, 1).to_pylist()[0]

        expected_state = np.concatenate(
            [
                data["palm_pose_xyz_rpy_states"][frame_idx],
                [data["gripper_value_states"][frame_idx]],
                data["biotac_lh"][frame_idx][BIOTAC_CHANNELS],
            ]
        ).astype(np.float32)
        expected_actions = np.concatenate(
            [data["palm_pose_xyz_rpy_commands"][frame_idx], [data["gripper_value_commands"][frame_idx]]]
        ).astype(np.float32)
        got_state = np.asarray(row["state"], dtype=np.float32)
        got_actions = np.asarray(row["actions"], dtype=np.float32)
        if not np.allclose(got_state, expected_state, atol=1e-4):
            errors.append(f"episode {ep_idx} frame {local_idx}: state mismatch got={got_state} expected={expected_state}")
        if not np.allclose(got_actions, expected_actions, atol=1e-4):
            errors.append(f"episode {ep_idx} frame {local_idx}: actions mismatch got={got_actions} expected={expected_actions}")

        for key in IMAGE_KEYS:
            png_bytes = row[key]["bytes"]
            got_img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
            expected_img = np.asarray(data[key][frame_idx])
            if got_img.shape != expected_img.shape:
                errors.append(f"episode {ep_idx} frame {local_idx}: {key} shape mismatch got={got_img.shape} expected={expected_img.shape}")
                continue
            diff = np.abs(got_img.astype(np.float32) - expected_img.astype(np.float32)).mean()
            if diff > image_diff_threshold:
                errors.append(f"episode {ep_idx} frame {local_idx}: {key} mean abs pixel diff={diff:.2f} > threshold={image_diff_threshold}")

        if "env_image_valid" in row:
            expected_valid = int(np.any(data["env_image"][s:e]))
            got_valid = int(np.asarray(row["env_image_valid"]).ravel()[0])
            if got_valid != expected_valid:
                errors.append(
                    f"episode {ep_idx} frame {local_idx}: env_image_valid={got_valid} but "
                    f"source episode env_image is {'non-zero' if expected_valid else 'all-zero'}"
                )
        checked += 1

    # --- Report ---
    print(f"\nChecked {info['total_episodes']} episodes / {info['total_frames']} frames.")
    print(f"Spot-checked {checked} random frames against source zarr.")
    print(f"Scanned {total_rows_scanned} rows for NaN/Inf in state/actions.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\nFAILED: {len(errors)} issue(s) found:")
        for err in errors[:50]:
            print(f"  - {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        raise SystemExit(1)

    print("\nAll correctness checks passed (see warnings above for data-quality caveats).")


if __name__ == "__main__":
    tyro.cli(main)
