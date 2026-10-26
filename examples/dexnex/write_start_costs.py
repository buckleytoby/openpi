"""Each frame's start-state task cost -> <dataset>/meta/frame_start_cost.npy (float32, one per frame in the dataset's
order; NaN for frames without one, e.g. the real data's): what DataConfig.cost_bins balances the sim sources by.

The costs come per sim zarr, in its frame order (avatar_intelligence: tree_to_zarr.start_costs of each frame's tree node
-- e.g. the <zarr name>_start_cost.npy files beside the zarrs). Each source's episodes are the k-th episode of that label
<-> the zarr's episode k (meta/episode_sources.json; appends keep the zarr's order); every episode's length is checked.

HF_LEROBOT_HOME=~/datasets uv run --no-sync python -m examples.dexnex.write_start_costs --repo-name dexnex/task_24_erlenmeyer_flask_cotrain \
    --costs ~/datasets/aiet_grip_sim_to_real.zarr=~/datasets/aiet_grip_sim_to_real_start_cost.npy \
            ~/datasets/aiet_grip_lift_sim_to_real.zarr=~/datasets/aiet_grip_lift_sim_to_real_start_cost.npy
"""

import json
import os

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import numpy as np
import tyro
import zarr

from examples.dexnex.convert_dexnex_data_to_lerobot import source_label


def main(repo_name: str, costs: list[str]) -> None:
    output_path = HF_LEROBOT_HOME / repo_name
    labels = json.loads((output_path / "meta" / "episode_sources.json").read_text())
    lengths = [json.loads(line)["length"] for line in (output_path / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    frame_cost = np.full(int(sum(lengths)), np.nan, np.float32)
    for item in costs:
        zarr_path, cost_path = (os.path.expanduser(p) for p in item.split("="))
        label = source_label(zarr_path, 1)
        values = np.load(cost_path)
        ends = np.asarray(zarr.open(zarr_path, mode="r")["meta/episode_ends"][:])
        zstarts = np.concatenate([[0], ends[:-1]])
        mine = [i for i, l in enumerate(labels) if l == label]
        if len(mine) != len(ends) or len(values) != ends[-1]:
            raise SystemExit(f"{label}: {len(mine)} episodes in the dataset, {len(ends)} in {zarr_path}; {len(values)} costs for {ends[-1]} frames")
        for k, i in enumerate(mine):
            n = int(ends[k] - zstarts[k])
            if n != lengths[i]:
                raise SystemExit(f"{label}: dataset episode {i} has {lengths[i]} frames, {zarr_path}'s episode {k} {n}")
            frame_cost[starts[i]:starts[i] + n] = values[zstarts[k]:ends[k]]
        print(f"{label}: {len(mine)} episodes, {int(ends[-1])} frames, start cost median {np.median(values):.2f}")
    np.save(output_path / "meta" / "frame_start_cost.npy", frame_cost)
    print(f"-> {output_path / 'meta' / 'frame_start_cost.npy'}: {int(np.isfinite(frame_cost).sum())} of {len(frame_cost)} frames with a cost")


if __name__ == "__main__":
    tyro.cli(main)
