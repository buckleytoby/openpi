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

meta/episode_sources.json: each episode's source label, by episode index -- "real" for --data_path's, a sim zarr's its
file name without "_sim_to_real.zarr" (aiet_grip_sim_to_real.zarr: "aiet_grip"). The training side's per-source
batch shares (openpi DataConfig.source_weights) read it. A dataset made before that file existed: filled in from each
episode's is_sim when it has a single sim source, on the next --append.

--refresh-actions: the sim episodes already in the dataset get their `actions` rewritten from their zarrs' current
commands (avatar_intelligence's sim_to_real relabel changes them in place: consensus commands) -- nothing else touched;
with --append, after the new episodes are added.

--append: add only what's new to an existing dataset instead of rebuilding it -- the episodes of each zarr past those
already converted (meta/converted_sources.json: per source zarr, its episodes done; zarrs are append-only, so new
episodes are the last ones). A dataset made before that file existed: inferred (the real zarr's episodes first, then
the sim ones), printed. Then recompute the norm stats (scripts/compute_norm_stats.py).

If you want to push your dataset to the Hugging Face Hub, add --push_to_hub.

The resulting dataset is saved to $HF_LEROBOT_HOME / <repo_name> (defaults to
~/.cache/huggingface/lerobot / <repo_name> if HF_LEROBOT_HOME is not set -- set it to
point at the SSD as shown above, since the converted dataset can be large).
"""

import concurrent.futures
import os
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
    append: bool = False,
    refresh_actions: bool = False,
    refresh_images: bool = False,
    workers: int = 0,
    push_to_hub: bool = False,
):
    output_path = HF_LEROBOT_HOME / repo_name
    sources = [(data_path, 0, 1)] + [(path, 1, sim_repeat) for path in (sim_paths or [])]
    if refresh_actions and not append:  # (only the existing sim episodes' actions: e.g. after sim_to_real's relabel)
        _refresh_actions(output_path)
        return
    if refresh_images:  # (only the existing sim episodes' images: after avatar_intelligence's run_rerender_images.sh)
        _refresh_images(output_path, workers)  # (workers: processes, 0 = one per CPU)
        return
    if append and output_path.exists():
        dataset = _open_for_append(repo_name)
        done = _converted_sources(output_path, dataset.meta.total_episodes, sources)
        labels = _episode_sources(output_path, dataset.meta.total_episodes, done, data_path)
        totals = {0: 0, 1: 0}
        for path, is_sim, repeat in sources:
            if not os.path.exists(path) and path in done:  # (converted before, its zarr not on this machine -- e.g. the real data's: nothing new from it)
                print(f"{path}: not here; its {done[path]} converted episodes kept, nothing appended from it")
                continue
            for _ in range(repeat):
                totals[is_sim] += _write_zarr(dataset, path, task, is_sim, start_episode=done.get(path, 0), labels=labels)
            done[path] = _num_episodes(path)
            _save_converted(output_path, done, labels)
        if refresh_actions:  # (the episodes converted before this append: their actions from the zarrs' current commands)
            _refresh_actions(output_path)
        _summary(output_path, totals, appended=True)
        return

    # Clean up any existing dataset in the output directory
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

    totals = {0: 0, 1: 0}
    done, labels = {}, []
    for path, is_sim, repeat in sources:
        for _ in range(repeat):
            totals[is_sim] += _write_zarr(dataset, path, task, is_sim, labels=labels)
        done[path] = _num_episodes(path)
        _save_converted(output_path, done, labels)
    _summary(output_path, totals, appended=False)

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


def _summary(output_path, totals: dict, appended: bool) -> None:
    """The run's summary: frames written now (real / sim), and the whole dataset's real / sim episodes and frames (its
    episodes in the order this script writes them: each source zarr's, as meta/converted_sources.json records)."""
    import json

    lengths = [json.loads(line)["length"] for line in open(output_path / "meta" / "episodes.jsonl")]
    done = json.loads((output_path / "meta" / "converted_sources.json").read_text())
    real_episodes = next(iter(done.values()), 0)  # (the real zarr: written first)
    real, sim = lengths[:real_episodes], lengths[real_episodes:]
    total = max(sum(lengths), 1)
    print(f"\n==== LeRobot dataset {'append' if appended else 'build'}: {output_path}")
    print(f"written now: {totals[0]} real frames, {totals[1]} sim frames")
    print(f"the dataset: {len(lengths)} episodes, {sum(lengths)} frames")
    print(f"  real {len(real):5d} episodes {sum(real):7d} frames ({100 * sum(real) / total:.1f}%)")
    print(f"  sim  {len(sim):5d} episodes {sum(sim):7d} frames ({100 * sum(sim) / total:.1f}%)  -- real : sim = "
          f"{sum(real) / max(sum(sim), 1):.2f} : 1")
    for path, count in done.items():
        print(f"  from {path}: {count} episodes")
    record = output_path / "meta" / "episode_sources.json"
    if record.exists():
        labels = json.loads(record.read_text())
        for label in dict.fromkeys(labels):
            frames = sum(n for n, l in zip(lengths, labels) if l == label)
            print(f"  source {label!r}: {labels.count(label)} episodes, {frames} frames ({100 * frames / total:.1f}%)")
    print("next: recompute the norm stats (scripts/compute_norm_stats.py), then rsync the dataset and assets to Quest")
    print("====", flush=True)


def _refresh_actions(output_path) -> None:
    """Every sim episode's `actions` (palm commands + gripper command) rewritten from its source zarr's current commands --
    e.g. after avatar_intelligence's sim_to_real relabel (consensus commands) changed them in place; nothing else
    touched (the images, states and metadata stay). A source's episodes are in the dataset in its zarr's order
    (meta/episode_sources.json: their label), each episode's rows checked against the zarr's. Then recompute the norm
    stats and rsync (the changed parquet files)."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    labels = json.loads((output_path / "meta" / "episode_sources.json").read_text())
    done = json.loads((output_path / "meta" / "converted_sources.json").read_text())
    chunks_size = json.loads((output_path / "meta" / "info.json").read_text()).get("chunks_size", 1000)
    for path in list(done)[1:]:  # (the real zarr is the first source: unchanged)
        label = source_label(path, 1)
        episodes = [i for i, l in enumerate(labels) if l == label]
        root = zarr.open(path, mode="r")
        ends = np.asarray(root["meta/episode_ends"])
        palm = np.asarray(root["data/palm_pose_xyz_rpy_commands"])
        grip = np.asarray(root["data/gripper_value_commands"])
        changed, total = 0, 0.0
        for k, episode in enumerate(episodes):
            start = int(ends[k - 1]) if k > 0 else 0
            end = int(ends[k])
            file = output_path / "data" / f"chunk-{episode // chunks_size:03d}" / f"episode_{episode:06d}.parquet"
            table = pq.read_table(file)
            if table.num_rows != end - start:
                raise SystemExit(f"{file}: {table.num_rows} rows, {path}'s episode {k} has {end - start} -- not this zarr's order")
            actions = np.concatenate([palm[start:end], grip[start:end, None]], axis=1).astype(np.float32)
            old = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
            total += float(np.abs(old - actions).mean())
            column = pa.FixedSizeListArray.from_arrays(pa.array(actions.reshape(-1), type=pa.float32()), 7)
            table = table.set_column(table.schema.get_field_index("actions"), table.schema.field("actions"), column)
            pq.write_table(table, file)
            changed += 1
        print(f"{label}: {changed} episodes' actions refreshed from {path} (mean |change| {total / max(changed, 1):.4f})")
    print("next: recompute the norm stats (scripts/compute_norm_stats.py), then rsync the dataset and assets to Quest")


def _refresh_episode(job) -> str:
    """(a _refresh_images worker) One episode's parquet with its camera images replaced by its source zarr's frames
    [start, end), written to a temporary file then moved over the old one (an interrupted refresh never leaves a
    half-written episode). Returns an error message, or ""."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    file, path, k, start, end = job
    table = pq.read_table(file)
    if table.num_rows != end - start:
        return f"{file}: {table.num_rows} rows, {path}'s episode {k} has {end - start} -- not this zarr's order"
    data = zarr.open(path, mode="r")["data"]

    def png(frame: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="PNG")
        return buffer.getvalue()

    for key in ("wrist_image", "env_image", "overhead_image"):
        frames = data[key][start:end]
        old = table.column(key)
        paths = [v["path"] for v in old.to_pylist()]
        column = pa.StructArray.from_arrays([pa.array([png(f) for f in frames], type=pa.binary()), pa.array(paths, type=pa.string())],
                                            fields=list(old.type))
        table = table.set_column(table.schema.get_field_index(key), table.schema.field(key), column)
    tmp = file.with_name(file.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, file)
    return ""


def _refresh_images(output_path, workers: int = 0) -> None:
    """Every sim episode's camera images (wrist / env / overhead) rewritten from its source zarr's -- after
    avatar_intelligence's run_rerender_images.sh re-rendered them there (e.g. domain randomization); only the episodes
    the zarr records as patched (attrs "images_patched"). Each episode's rows checked against the zarr's; the states,
    actions and metadata stay (the norm stats are the states' and actions': unchanged). ``workers`` processes (one
    episode each at a time; 0: one per CPU)."""
    import json
    import multiprocessing

    labels = json.loads((output_path / "meta" / "episode_sources.json").read_text())
    done = json.loads((output_path / "meta" / "converted_sources.json").read_text())
    chunks_size = json.loads((output_path / "meta" / "info.json").read_text()).get("chunks_size", 1000)
    workers = workers or os.cpu_count() or 1

    for path in list(done)[1:]:  # (the real zarr is the first source: unchanged)
        if not os.path.exists(path):
            print(f"{path}: not here -- skipped")
            continue
        label = source_label(path, 1)
        episodes = [i for i, l in enumerate(labels) if l == label]
        root = zarr.open(path, mode="r")
        patched = set(root.attrs.get("images_patched", []))
        trees = [os.path.basename(str(t)) for t in root.attrs["sim_episodes"]]
        ends = np.asarray(root["meta/episode_ends"])
        jobs = []
        for k, episode in enumerate(episodes):
            if k >= len(trees) or trees[k] not in patched:
                continue
            file = output_path / "data" / f"chunk-{episode // chunks_size:03d}" / f"episode_{episode:06d}.parquet"
            jobs.append((file, path, k, int(ends[k - 1]) if k > 0 else 0, int(ends[k])))
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            errors = [m for m in tqdm.tqdm(pool.imap_unordered(_refresh_episode, jobs), total=len(jobs), desc=f"{label}: images ({workers} workers)") if m]
        if errors:
            raise SystemExit("\n".join(errors[:10]))
        print(f"{label}: {len(jobs)} episodes' images refreshed from {path} ({len(patched)} patched there)")
    print("next: rsync the dataset (the rewritten parquet files) to where it trains")


def _open_for_append(repo_name: str):
    """An existing LeRobot dataset, ready for add_frame / save_episode -- LeRobotDataset.create's setup on the existing
    metadata, without loading its data (LeRobotDataset(repo) would build the whole Arrow cache first)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.video_utils import get_safe_default_codec

    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = LeRobotDatasetMetadata(repo_name, HF_LEROBOT_HOME / repo_name)
    obj.repo_id, obj.root, obj.revision, obj.tolerance_s = obj.meta.repo_id, obj.meta.root, None, 1e-4
    obj.image_writer = None
    obj.start_image_writer(5, 10)
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes, obj.image_transforms, obj.delta_timestamps, obj.delta_indices, obj.episode_data_index = None, None, None, None, None
    obj.hf_dataset = obj.create_hf_dataset()
    obj.video_backend = get_safe_default_codec()
    return obj


def _num_episodes(path: str) -> int:
    return len(np.asarray(zarr.open(path, mode="r")["meta/episode_ends"]))


def _save_converted(output_path, done: dict, labels: list) -> None:
    import json

    (output_path / "meta" / "converted_sources.json").write_text(json.dumps(done, indent=1))
    (output_path / "meta" / "episode_sources.json").write_text(json.dumps(labels))


def source_label(path: str, is_sim: int) -> str:
    """An episode's source label (meta/episode_sources.json): "real", or the sim zarr's name without
    "_sim_to_real.zarr" / ".zarr"."""
    import pathlib

    if not is_sim:
        return "real"
    name = pathlib.Path(path.rstrip("/")).name
    return name.removesuffix(".zarr").removesuffix("_sim_to_real")


def _episode_sources(output_path, total_episodes: int, done: dict, data_path: str) -> list:
    """Each existing episode's source label: meta/episode_sources.json, else (a dataset from before that file) from each
    episode's is_sim -- possible while it has a single sim source (converted_sources.json)."""
    import json

    import pyarrow.parquet as pq

    record = output_path / "meta" / "episode_sources.json"
    if record.exists():
        labels = json.loads(record.read_text())
        if len(labels) != total_episodes:
            raise SystemExit(f"{record}: {len(labels)} labels for {total_episodes} episodes")
        return labels
    sims = [path for path, count in done.items() if path != data_path and count > 0]
    if len(sims) > 1:
        raise SystemExit(f"no {record.name} and {len(sims)} sim sources ({sims}): which sim episode is which is unknown; rebuild instead (no --append)")
    sim_label = source_label(sims[0], 1) if sims else None
    labels = []
    for episode in range(total_episodes):
        files = list((output_path / "data").glob(f"chunk-*/episode_{episode:06d}.parquet"))
        value = pq.read_table(files[0], columns=["is_sim"]).column("is_sim")[0].as_py()
        is_sim = int(value[0] if isinstance(value, list) else value)
        labels.append(sim_label if is_sim else "real")
    counts = {label: labels.count(label) for label in dict.fromkeys(labels)}
    print(f"no {record.name}: filled in from is_sim -- {counts}")
    return labels


def _converted_sources(output_path, total_episodes: int, sources: list) -> dict:
    """Per source zarr, its episodes already in the dataset: meta/converted_sources.json, else inferred (the sources in
    order, each fully converted while the dataset's episode count allows -- how this script wrote them)."""
    import json

    record = output_path / "meta" / "converted_sources.json"
    if record.exists():
        return json.loads(record.read_text())
    done, left = {}, total_episodes
    for path, _, repeat in sources:
        count = min(_num_episodes(path), left // max(repeat, 1))
        done[path] = count
        left -= count * repeat
    print(f"no {record.name}: inferred from the dataset's {total_episodes} episodes -- {done}")
    if left:
        raise SystemExit(f"{left} episodes unaccounted for: not this script's real + sim layout; rebuild instead (no --append)")
    return done


def _write_zarr(dataset, path: str, task: str, is_sim: int, start_episode: int = 0, labels: list | None = None) -> int:
    """The episodes of the zarr at ``path`` from ``start_episode`` on into ``dataset`` (one LeRobot episode each; each
    one's source label appended to ``labels``); returns their frame count."""
    data, episode_ends = _open_zarr(path)
    start = int(episode_ends[start_episode - 1]) if start_episode > 0 else 0
    num_steps = int(episode_ends[-1]) - start
    if num_steps <= 0:
        print(f"{path}: nothing new ({start_episode} episodes already converted)")
        return 0
    def load(a: int, b: int) -> dict:  # (an episode's frames: the images' JPEG 2000 decoded -- ~25% of an episode's time)
        return {key: data[key][a:b] for key in ("palm_pose_xyz_rpy_states", "gripper_value_states", "biotac_lh", "palm_pose_xyz_rpy_commands",
                                                 "gripper_value_commands", "wrist_image", "env_image", "overhead_image")}

    spans = []  # (episode index, start, end) of the episodes to convert
    for episode_idx, end in enumerate(episode_ends):
        if episode_idx >= start_episode:
            spans.append((episode_idx, start, int(end)))
            start = int(end)
    # the next episode read in a background thread while this one is saved (save_episode: the PNGs, the parquet, the stats)
    prefetch = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    with tqdm.tqdm(total=num_steps, desc=f"Converting {'sim' if is_sim else 'real'} frames", unit="frame") as pbar:
        nxt = prefetch.submit(load, spans[0][1], spans[0][2])
        for k, (episode_idx, start, end) in enumerate(spans):
            frames = nxt.result()
            if k + 1 < len(spans):
                nxt = prefetch.submit(load, spans[k + 1][1], spans[k + 1][2])
            palm_states = frames["palm_pose_xyz_rpy_states"]
            gripper_states = frames["gripper_value_states"]
            biotac = frames["biotac_lh"][:, BIOTAC_CHANNELS]
            palm_commands = frames["palm_pose_xyz_rpy_commands"]
            gripper_commands = frames["gripper_value_commands"]
            wrist_images = frames["wrist_image"]
            env_images = frames["env_image"]
            overhead_images = frames["overhead_image"]

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
            if labels is not None:
                labels.append(source_label(path, is_sim))
            # LeRobot keeps every saved episode, images embedded, in dataset.hf_dataset (save_episode's
            # concatenate_datasets): ~38 GB of RAM by 230k frames (the real + sim dataset got OOM-killed). Each
            # episode is already on disk as its parquet, and nothing here reads it back: start it empty again.
            dataset.hf_dataset = dataset.create_hf_dataset()
    prefetch.shutdown()
    return num_steps


if __name__ == "__main__":
    tyro.cli(main)
