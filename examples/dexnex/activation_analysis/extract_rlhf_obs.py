"""Held-out real observations for the pi0.5 activation analysis: RLHF decisions (none of the checkpoints trained on
them), as the openpi node saw them at the decision.

Per sampled decision (openpi's base path voted up or down, every camera recorded in its session's bag):
  images   the three cameras' first frame at or after t_decision (the bags' decimated, compressed copies of the node's
           topics -- JPEG on top of the training images' raw frames), resized to 224 as the node does
  state    [10] the node's observation/state: palm pose + gripper (the record's live_palm_gripper) + BioTac channels
           [0, 1, 4] (the record's biotac)
  base     [6, 7] the executed base path's palm + gripper targets (openpi's chunk, steps 4..9: n_waypoints_skip 4, keep 6)
  label    up / down; session, seq, base_note (the checkpoint that ran)

Needs ROS 2 (rosbag2_py): the system Python.

    /usr/bin/python3 extract_rlhf_obs.py --out /media/dexnex_ssd/data/activation_analysis/rlhf_obs.npz --per-label 400
"""
import argparse
import collections
import glob
import io
import json
import os

import numpy as np
import yaml
from PIL import Image

SESSIONS = "/media/dexnex_ssd/data/rlhf/*/"
CAMERAS = {  # (the node's observation key -> the bag's topic)
    "wrist_image": "/avatar/left_arm/wrist_cam/decimated/compressed/temporal_throttle",
    "env_image": "/avatar/env/yz_cam/decimated/compressed/temporal_throttle",
    "overhead_image": "/dexnex_projects_common/leftcam/image_resized/throttled/compressed/temporal_throttle",
}
BIOTAC_CHANNELS = [0, 1, 4]
MAX_LAG = 0.5  # [s] a camera frame this long after the decision at most


def resize_224(img: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))


def bag_of(session: str) -> str | None:
    meta = glob.glob(session + "rlhf_*/metadata.yaml")
    if not meta:
        return None
    info = yaml.safe_load(open(meta[0]))["rosbag2_bagfile_information"]
    counts = {t["topic_metadata"]["name"]: t["message_count"] for t in info["topics_with_message_count"]}
    if not all(counts.get(t, 0) > 0 for t in CAMERAS.values()):
        return None
    return os.path.dirname(meta[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per-label", type=int, default=400, help="decisions voted up, and as many down")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    rows = []  # (session, bag, record)
    for d in sorted(glob.glob(SESSIONS)):
        bag = bag_of(d)
        if bag is None or not os.path.exists(d + "sequences.jsonl"):
            continue
        for line in open(d + "sequences.jsonl"):
            r = json.loads(line)
            if (r.get("source") == "openpi" and r.get("label") in ("up", "down") and r.get("live_palm_gripper") is not None
                    and r.get("biotac") is not None and r.get("base") is not None):
                rows.append((os.path.basename(d.rstrip("/")), bag, r))
    rng = np.random.default_rng(args.seed)
    picked = []
    for label in ("up", "down"):
        idx = [i for i, (_, _, r) in enumerate(rows) if r["label"] == label]
        picked += [rows[i] for i in rng.choice(idx, min(args.per_label, len(idx)), replace=False)]
    by_bag = collections.defaultdict(list)
    for row in picked:
        by_bag[row[1]].append(row)
    print(f"{len(rows)} voted openpi decisions with all cameras; {len(picked)} sampled from {len(by_bag)} sessions", flush=True)

    out = collections.defaultdict(list)
    for bag, items in sorted(by_bag.items()):
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(CAMERAS.values())))
        stamps = {t: [] for t in CAMERAS.values()}
        raws = {t: [] for t in CAMERAS.values()}
        while reader.has_next():
            topic, raw, t = reader.read_next()
            stamps[topic].append(t / 1e9)
            raws[topic].append(raw)
        stamps = {t: np.asarray(v) for t, v in stamps.items()}
        kept = 0
        for session, _, r in items:
            want = r.get("t_decision", r["t_start"])
            frames = {}
            for key, topic in CAMERAS.items():
                j = int(np.searchsorted(stamps[topic], want))
                if j >= len(stamps[topic]) or stamps[topic][j] - want > MAX_LAG:
                    break
                msg = deserialize_message(raws[topic][j], CompressedImage)
                frames[key] = resize_224(np.asarray(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")))
            if len(frames) < len(CAMERAS):
                continue
            base = np.asarray(r["base"], np.float32)
            if base.shape != (6, 7):
                continue
            biotac = np.asarray(r["biotac"], np.float32)[BIOTAC_CHANNELS]
            for key in CAMERAS:
                out[key].append(frames[key])
            out["state"].append(np.concatenate([np.asarray(r["live_palm_gripper"], np.float32), biotac]))
            out["base"].append(base)
            out["label"].append(r["label"])
            out["session"].append(session)
            out["seq"].append(int(r["seq"]))
            out["base_note"].append(r.get("base_note") or "")
            kept += 1
        print(f"  {os.path.basename(bag)}: {kept}/{len(items)} decisions", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **{k: np.asarray(v) for k, v in out.items()})
    print(f"{args.out}: {len(out['label'])} decisions ({collections.Counter(out['label'])})", flush=True)


if __name__ == "__main__":
    main()
