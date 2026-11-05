"""Each frame's views -> <dataset>/meta/frame_views.npz (one bool array per view, one entry per frame in the dataset's order;
false for the real data's frames -- only sim frames are tagged -- except --real-views): what DataConfig.view_weights samples more.

The views come per sim zarr, in its frame order (avatar_intelligence tree_exploration/sim_views.py: <zarr name>_views.npz
beside the zarrs -- flush_corner, fine_align). Each source's episodes are the k-th episode of that label <-> the zarr's
episode k (meta/episode_sources.json; appends keep the zarr's order); every episode's length is checked.

HF_LEROBOT_HOME=~/datasets uv run --no-sync python -m examples.dexnex.write_frame_views --repo-name dexnex/task_24_erlenmeyer_flask_cotrain \
    --views ~/datasets/aiet_grip_sim_to_real.zarr=~/datasets/aiet_grip_sim_to_real_views.npz \
            ~/datasets/aiet_grip_lift_sim_to_real.zarr=~/datasets/aiet_grip_lift_sim_to_real_views.npz
"""

import json
import os

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import numpy as np
import tyro
import zarr

from examples.dexnex.convert_dexnex_data_to_lerobot import source_label

VIEWS = ("flush_corner", "fine_align")


def main(repo_name: str, views: list[str], real_views: list[str] = []) -> None:
    """``real_views``: NAME=K items -- a view of every frame of the real episodes from the K-th real episode on (0-based,
    in dataset order: e.g. K = the real episode count before an --append of new real bags)."""
    output_path = HF_LEROBOT_HOME / repo_name
    labels = json.loads((output_path / "meta" / "episode_sources.json").read_text())
    lengths = [json.loads(line)["length"] for line in (output_path / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    frame_views = {view: np.zeros(int(sum(lengths)), bool) for view in VIEWS}
    real = [i for i, l in enumerate(labels) if l == "real"]
    for item in real_views:
        name, first = item.split("=")
        if name in VIEWS:
            raise SystemExit(f"--real-views {name}: a sim view's name")
        tags = frame_views[name] = np.zeros(int(sum(lengths)), bool)
        for i in real[int(first):]:
            tags[starts[i]:starts[i] + lengths[i]] = True
        print(f"real view {name}: real episodes {first}..{len(real) - 1} ({len(real[int(first):])} episodes, {int(tags.sum())} frames)")
        if not tags.any():
            raise SystemExit(f"--real-views {item}: no real episodes from {first} on ({len(real)} real episodes)")
    for item in views:
        zarr_path, views_path = (os.path.expanduser(p) for p in item.split("="))
        label = source_label(zarr_path, 1)
        tags = np.load(views_path)
        ends = np.asarray(zarr.open(zarr_path, mode="r")["meta/episode_ends"][:])
        zstarts = np.concatenate([[0], ends[:-1]])
        mine = [i for i, l in enumerate(labels) if l == label]
        if len(mine) != len(ends) or any(len(tags[view]) != ends[-1] for view in VIEWS):
            raise SystemExit(f"{label}: {len(mine)} episodes in the dataset, {len(ends)} in {zarr_path}; "
                             f"{[len(tags[view]) for view in VIEWS]} tagged frames for {ends[-1]}")
        for k, i in enumerate(mine):
            n = int(ends[k] - zstarts[k])
            if n != lengths[i]:
                raise SystemExit(f"{label}: dataset episode {i} has {lengths[i]} frames, {zarr_path}'s episode {k} {n}")
            for view in VIEWS:
                frame_views[view][starts[i]:starts[i] + n] = tags[view][zstarts[k]:ends[k]]
        print(f"{label}: {len(mine)} episodes, {int(ends[-1])} frames | " + ", ".join(f"{view} {int(tags[view].sum())}" for view in VIEWS))
    np.savez(output_path / "meta" / "frame_views.npz", **frame_views)
    print(f"-> {output_path / 'meta' / 'frame_views.npz'}: " + ", ".join(
        f"{view} {int(v.sum())} of {len(v)} frames ({100 * v.mean():.1f}%)" for view, v in frame_views.items()))


if __name__ == "__main__":
    tyro.cli(main)
