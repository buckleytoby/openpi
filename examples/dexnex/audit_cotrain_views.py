"""Audit a view-weighted co-train dataset before training (pi05_dexnex_erlenmeyer_cotrain_balanced_rot_costbal_views_layout):
everything the sampler and the sim data need is in, and the sim episodes are the layout-randomized, domain-randomized ones.

  sources        meta/episode_sources.json / converted_sources.json: each source's episodes and frames; the sim zarrs the
                 layout ones (OUT_DIR "layout"), every zarr's episode count matching the dataset's
  start costs    meta/frame_start_cost.npy: one per frame, finite on every sim frame, NaN on the real ones; frames per
                 costbal bin
  views          meta/frame_views.npz: flush_corner / fine_align per frame, none on real frames
  layout         (not recorded per frame: the layout draws lived in the deleted episode npzs) the trees in both the old and
                 the new sim zarr: the same path re-rendered under a random layout comes out rigidly shifted in the robot
                 frame -- a different offset per episode (the table +-5 cm / 5 deg, the tray +-2 cm / 3 deg); none unshifted
  domain rand.   (likewise) the overhead camera's mean colour across episodes vs the old zarr's: the randomized looks widen it
  norm stats     the config's norm_stats.json exists
  sampler        the config's sampler built for real (data_loader.source_weighted_sampler): each source's and each view's
                 share of a batch

HF_LEROBOT_HOME=~/datasets uv run --no-sync python -m examples.dexnex.audit_cotrain_views \
    --old-zarrs ~/datasets/aiet_grip_sim_to_real.zarr ~/datasets/aiet_grip_lift_sim_to_real.zarr
"""

import collections
import json
import os

import imagecodecs.numcodecs as _imagecodecs_numcodecs
import numpy as np
import tyro
import zarr

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

_imagecodecs_numcodecs.register_codecs()

OK, BAD = "OK  ", "FAIL"


def _episode_summary(zarr_path: str, n_episodes: int = 120, seed: int = 0):
    """(each episode's first palm xyz [n, 3], a sample of episodes' overhead mean colours [k, 3])."""
    root = zarr.open(os.path.expanduser(zarr_path), mode="r")
    ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    palm = np.asarray(root["data/palm_pose_xyz_rpy_states"].get_orthogonal_selection((starts, slice(0, 3))))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(starts), size=min(n_episodes, len(starts)), replace=False)
    colours = np.stack([np.asarray(root["data/overhead_image"][int(starts[k] + (ends[k] - starts[k]) // 2)]).reshape(-1, 3).mean(0) for k in pick])
    return palm, colours


def _matched_shifts(old: str, new: str, limit: int = 200):
    """For up to ``limit`` trees in both zarrs (sim_episodes, by file name): the mean palm x / y shift [m] and yaw shift [deg]
    of the new episode vs the old."""
    zo, zn = zarr.open(os.path.expanduser(old), mode="r"), zarr.open(os.path.expanduser(new), mode="r")

    def index(z):
        e = np.asarray(z["meta/episode_ends"][:])
        s = np.concatenate([[0], e[:-1]])
        return {os.path.basename(p): (int(a), int(b)) for p, a, b in zip(z.attrs.get("sim_episodes", []), s, e)}

    io, inn = index(zo), index(zn)
    offsets, yaws = [], []
    for t in sorted(set(io) & set(inn))[:limit]:
        (a, b), (c, d) = io[t], inn[t]
        if b - a != d - c:
            continue
        po = np.asarray(zo["data/palm_pose_xyz_rpy_states"][a:b])
        pn = np.asarray(zn["data/palm_pose_xyz_rpy_states"][c:d])
        offsets.append(float(np.linalg.norm((pn[:, :2] - po[:, :2]).mean(0))))
        yaws.append(float(np.degrees(np.angle(np.exp(1j * (pn[:, 5] - po[:, 5]))).mean())))
    return np.array(offsets), np.array(yaws)


def main(
    repo_name: str = "dexnex/task_24_erlenmeyer_flask_cotrain_layout",
    config_name: str = "pi05_dexnex_erlenmeyer_cotrain_balanced_rot_costbal_views_layout",
    old_zarrs: list[str] | None = None,
) -> None:
    root = HF_LEROBOT_HOME / repo_name
    meta = root / "meta"
    results = []

    def check(ok: bool, what: str) -> None:
        results.append(ok)
        print(f"[{OK if ok else BAD}] {what}")

    # sources
    labels = json.loads((meta / "episode_sources.json").read_text())
    lengths = [json.loads(l)["length"] for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
    converted = json.loads((meta / "converted_sources.json").read_text())
    n_frames = int(sum(lengths))
    print(f"\n== {repo_name}: {len(lengths)} episodes, {n_frames} frames")
    per = collections.Counter(labels)
    frames = collections.Counter()
    for l, n in zip(labels, lengths):
        frames[l] += n
    for label in per:
        print(f"   {label:16s} {per[label]:6d} episodes {frames[label]:9d} frames ({100 * frames[label] / n_frames:.1f}%)")
    check(set(per) == {"real", "aiet_grip", "aiet_grip_lift"}, f"sources are real + aiet_grip + aiet_grip_lift ({sorted(per)})")
    sim_paths = {p: n for p, n in converted.items() if "sim_to_real" in p}
    check(all("/layout/" in p for p in sim_paths), f"the sim zarrs are the layout ones: {list(sim_paths)}")
    for path, n in converted.items():
        label = "real" if "sim_to_real" not in path else os.path.basename(path.rstrip("/")).removesuffix(".zarr").removesuffix("_sim_to_real")
        z_eps = int(zarr.open(path, mode="r")["meta/episode_ends"].shape[0]) if os.path.exists(path) else None
        check(per.get(label) == n and (z_eps is None or z_eps == n),
              f"{label}: {per.get(label)} dataset episodes, {n} converted, {z_eps if z_eps is not None else '(zarr not here)'} in {path}")

    # start costs
    frame_label = np.concatenate([np.full(n, l, dtype=object) for n, l in zip(lengths, labels)])
    sim = frame_label != "real"
    cost_file = meta / "frame_start_cost.npy"
    check(cost_file.exists(), f"{cost_file.name} exists")
    if cost_file.exists():
        cost = np.load(cost_file)
        check(len(cost) == n_frames, f"start costs: {len(cost)} for {n_frames} frames")
        if len(cost) == n_frames:
            check(bool(np.isfinite(cost[sim]).all()), f"every sim frame has a start cost ({int(np.isfinite(cost[sim]).sum())}/{int(sim.sum())})")
            check(bool(np.isnan(cost[~sim]).all()), "no real frame has one")
            for label, edges in (("aiet_grip", (0.0, 7.5, 8.5, 9.5, 10.0, 10.5, 11.0, 11.5, 100.0)), ("aiet_grip_lift", (0.0, 9.5, 10.0, 10.5, 11.0, 11.5, 100.0))):
                c = cost[frame_label == label]
                print(f"       {label} frames per costbal bin {list(edges)}: {np.histogram(c, edges)[0].tolist()}")

    # views
    view_file = meta / "frame_views.npz"
    check(view_file.exists(), f"{view_file.name} exists")
    if view_file.exists():
        views = np.load(view_file)
        for name in ("flush_corner", "fine_align"):
            v = np.asarray(views[name], bool) if name in views.files else None
            check(v is not None and len(v) == n_frames, f"view {name}: {'missing' if v is None else f'{len(v)} frames'}")
            if v is not None and len(v) == n_frames:
                check(not v[~sim].any(), f"   {name}: no real frame tagged")
                by = {l: int(v[frame_label == l].sum()) for l in ("aiet_grip", "aiet_grip_lift")}
                check(all(by.values()), f"   {name}: {int(v.sum())} frames ({by})")

    # layout + domain randomization: the new sim zarrs vs the old ones
    if old_zarrs:
        print("\n== layout and domain randomization (new sim zarrs vs the old ones)")
        new_by_label = {os.path.basename(p.rstrip("/")).removesuffix(".zarr").removesuffix("_sim_to_real"): p for p in sim_paths}
        for old in old_zarrs:
            label = os.path.basename(old.rstrip("/")).removesuffix(".zarr").removesuffix("_sim_to_real")
            new = new_by_label.get(label)
            if new is None or not os.path.exists(os.path.expanduser(old)):
                print(f"   {label}: no new / old zarr to compare")
                continue
            (_, c_old), (_, c_new) = _episode_summary(old), _episode_summary(new)
            offsets, yaws = _matched_shifts(old, new)
            if len(offsets):
                check(bool((offsets > 1e-3).all() and np.median(offsets) > 0.01),
                      f"{label} layout: {len(offsets)} trees in both zarrs, the same path's palm x / y shift (new vs old) 5/50/95 % "
                      f"{np.round(1000 * np.percentile(offsets, [5, 50, 95]), 1).tolist()} mm, {int((offsets <= 1e-3).sum())} unshifted; "
                      f"yaw 5/50/95 % {np.round(np.percentile(yaws, [5, 50, 95]), 1).tolist()} deg")
            else:
                print(f"   {label} layout: no tree in both zarrs to compare")
            v_old, v_new = c_old.std(0).mean(), c_new.std(0).mean()
            check(bool(v_new > 1.5 * v_old), f"{label} domain randomization: overhead mean colour spread across episodes {v_old:.1f} (old) -> {v_new:.1f} (new) /255")

    # norm stats + the sampler
    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader

    cfg = _config.get_config(config_name)
    stats = cfg.assets_dirs / cfg.data.repo_id / "norm_stats.json"
    check(stats.exists(), f"norm stats: {stats}" + ("" if stats.exists() else " -- run scripts/compute_norm_stats.py --config-name " + config_name))
    print(f"\n== {config_name}: the sampler")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    sampler = _data_loader.source_weighted_sampler(data_config, n_frames)
    w = np.asarray(sampler.weights, np.float64)
    w /= w.sum()
    for label in per:
        print(f"   {label:16s} {100 * w[frame_label == label].sum():5.1f}% of each batch")
    if view_file.exists():
        for name, share in (data_config.view_weights or {}).items():
            v = np.asarray(views[name], bool)
            print(f"   view {name:12s} {100 * w[v].sum():5.1f}% of each batch (its own share {100 * share:.0f}% + its frames' base weight)")
    print(f"\n{sum(results)}/{len(results)} checks passed" + ("" if all(results) else " -- see FAIL above"))


if __name__ == "__main__":
    tyro.cli(main)
