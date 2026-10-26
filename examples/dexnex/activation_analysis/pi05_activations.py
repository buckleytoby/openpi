"""pi0.5 hidden states for real and sim observations (the activation analysis: report.py reads these).

Per checkpoint, the same observations:
  rlhf        held-out real: RLHF decisions (extract_rlhf_obs.py's npz) -- no checkpoint trained on them
  real        real teleop demos, frames from the co-train LeRobot dataset (every checkpoint trained on them)
  aiet_grip / aiet_grip_lift   sim frames from the same dataset (only the co-train checkpoints trained on them)
each through the policy's own input transforms (DexnexInputs, Normalize with the checkpoint's norm stats, the
tokenizer), then ONE forward pass of the whole model as in training (compute_loss): the observation's prefix (three
cameras' SigLIP tokens + the prompt with the state) and the action expert's suffix (the noisy chunk at t = 1: the same
fixed noise for every sample, so the action tokens differ only through the observation). Every transformer layer's
output is recorded (a Block that sows them -- patched in here, openpi's code untouched) and pooled per observation:
  env / wrist / overhead   the mean over that camera's 256 image tokens [layers, 2048]
  text                     the mean over the prompt + state tokens [layers, 2048]
  action                   the mean over the action expert's chunk tokens [layers, 1024]
For the rlhf decisions also the checkpoint's own predicted chunk (policy.infer, a fixed noise): steps 4..9, the ones
the RLHF node executed (report.py compares it with the voted-up and voted-down base paths).

    cd ~/dexnex/libraries/openpi
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run examples/dexnex/activation_analysis/pi05_activations.py \\
        --config pi05_dexnex_erlenmeyer_cotrain_balanced --checkpoint /media/dexnex_ssd/models/openpi_cotrain_balanced/27999 \\
        --name balanced_27999
"""
import argparse
import glob
import io
import json
import os
import pathlib
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import gemma

OUT = "/media/dexnex_ssd/data/activation_analysis"
DATASET = "/media/dexnex_ssd/datasets/dexnex/task_24_erlenmeyer_flask_cotrain"
PROMPT = "pick up the pellet from the tray and place it in the erlenmeyer flask"
SKIP, KEEP = 4, 6  # (the RLHF node's n_waypoints_skip / n_waypoints)
SIM_ZARRS = {"aiet_grip": "/media/dexnex_ssd/datasets/aiet_grip_sim_to_real.zarr",
             "aiet_grip_lift": "/media/dexnex_ssd/datasets/aiet_grip_lift_sim_to_real.zarr"}
TREE_DIRS = {"aiet_grip": ["/media/dexnex_ssd/data/aiet_grip_tree_data/local",
                           "/home/omnid/dexnex/ws_avatar_drake/src/avatar_master/avatar_intelligence/tree_data"],
             "aiet_grip_lift": ["/media/dexnex_ssd/data/aiet_grip_lift_tree_data/local"]}


# ---- every layer's output, sown (a copy of gemma.Block.__call__ + one sow; the scan stacks them over the layers) -----
class SowBlock(nn.Module):
    configs: tuple
    dropout: float = 0.0
    dropout_bdims: tuple = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = gemma.sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x
        attn = gemma.Attention(configs=self.configs, name="attn")
        pre_attn, gates = [], []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = gemma.RMSNorm(name=gemma._name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        xs = [gemma._gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        out, gates = [], []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = gemma.RMSNorm(name=gemma._name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = gemma.lora.FeedForward(features=config.width, hidden_dim=config.mlp_dim, name=gemma._name("mlp", i),  # noqa: PLW2901
                                           lora_config=config.lora_configs.get("ffn"))(x)
            out.append(x)
            gates.append(gate if x is not None else None)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [gemma._gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        self.sow("intermediates", "hidden", tuple(x.astype(jnp.float32) for x in xs if x is not None))
        return xs, kv_cache


class SowModule(gemma.Module):  # (gemma.Module with SowBlock and the sown collection stacked over the scanned layers;
    def setup(self):              #  a subclass: flax wraps setup at class creation -- a patched-in setup loses the submodules' names)
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        self.embedder = gemma.Embedder(vocab_size=gemma.PALIGEMMA_VOCAB_SIZE, embed_dim=self.configs[0].width, name="embedder")
        block_cls = nn.remat(SowBlock, prevent_cse=False, static_argnums=(5,), policy=jax.checkpoint_policies.nothing_saveable)
        self.layers = nn.scan(block_cls, variable_axes={"params": 0, "intermediates": 0}, split_rngs={"params": True, "dropout": True},
                              in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast), length=self.configs[0].depth,
                              )(configs=self.configs, dropout=self.dropout, dropout_bdims=self.dropout_bdims)
        self.final_norms = [gemma.RMSNorm(name=gemma._name("final_norm", i)) for i in range(len(self.configs))]


gemma.Module = SowModule  # (pi0 builds its backbone as gemma.Module(...))


# ---- observations --------------------------------------------------------------------------------------------------
def png(cell) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def start_kinds(source: str) -> list:
    """The sim zarr's episodes' start kinds, in order (= the dataset's episodes of that source, in order); cached."""
    import zarr

    cache = f"{OUT}/start_kinds_{source}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    trees = list(zarr.open(SIM_ZARRS[source], mode="r").attrs["sim_episodes"])
    kinds = []
    for tree in trees:
        kind = "?"
        for d in TREE_DIRS[source]:
            path = os.path.join(d, os.path.basename(str(tree)))
            if os.path.exists(path):
                with np.load(path, allow_pickle=True) as t:
                    kind = ("bank" if "start" in t.files and str(t["start"]) == "actor"
                            else str(t["start_kind"]) if "start_kind" in t.files and str(t["start_kind"]) else "ood")
                break
        kinds.append(kind)
    os.makedirs(OUT, exist_ok=True)
    json.dump(kinds, open(cache, "w"))
    return kinds


def dataset_frames(per_source: dict, seed: int) -> list:
    """Frames from the co-train LeRobot dataset: per source, frames from random episodes at random times."""
    import pyarrow.parquet as pq

    sources = json.load(open(f"{DATASET}/meta/episode_sources.json"))
    rng = np.random.default_rng(seed)
    kinds = {s: start_kinds(s) for s in SIM_ZARRS}
    out = []
    for source, n in per_source.items():
        eps = [i for i, s in enumerate(sources) if s == source]
        first = eps[0]
        for ep in rng.choice(eps, min(n, len(eps)), replace=False):
            f = glob.glob(f"{DATASET}/data/chunk-*/episode_{int(ep):06d}.parquet")[0]
            t = pq.read_table(f, columns=["wrist_image", "env_image", "overhead_image", "state"])
            k = int(rng.integers(len(t)))
            row = {c: t.column(c)[k].as_py() for c in t.column_names}
            out.append(dict(source=source, episode=int(ep), progress=k / max(len(t) - 1, 1),
                            kind=kinds[source][int(ep) - first] if source in kinds else "real",
                            obs={"observation/wrist_image": png(row["wrist_image"]), "observation/env_image": png(row["env_image"]),
                                 "observation/overhead_image": png(row["overhead_image"]),
                                 "observation/state": np.asarray(row["state"], np.float32), "observation/env_image_valid": True,
                                 "prompt": PROMPT}))
    return out


def rlhf_frames(path: str, n: int = 0) -> list:
    npz = np.load(path, allow_pickle=True)
    z = {k: npz[k] for k in npz.files}  # (each array decompressed once: indexing the npz decompresses it every time)
    if n:
        z = {k: v[:n] for k, v in z.items()}
    return [dict(source="rlhf", episode=-1, progress=np.nan, kind=str(z["label"][i]), label=str(z["label"][i]), base=z["base"][i],
                 session=str(z["session"][i]), base_note=str(z["base_note"][i]),
                 obs={"observation/wrist_image": z["wrist_image"][i], "observation/env_image": z["env_image"][i],
                      "observation/overhead_image": z["overhead_image"][i], "observation/state": z["state"][i].astype(np.float32),
                      "observation/env_image_valid": True, "prompt": PROMPT})
            for i in range(len(z["label"]))]


# ---- activations -----------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True, help="the output's name: <out>/acts_<name>.npz")
    p.add_argument("--rlhf", default=f"{OUT}/rlhf_obs.npz")
    p.add_argument("--rlhf-n", type=int, default=0, help="at most this many rlhf decisions (0: all)")
    p.add_argument("--real", type=int, default=600)
    p.add_argument("--sim", type=int, default=400, help="frames per sim source")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from flax import nnx
    from flax.nnx.bridge import variables as bv

    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config
    from openpi.training import config as _config

    print(f"devices: {jax.devices()}", flush=True)  # (the GPU set up first: after reading ~1,400 parquet files the page cache
    # leaves little free memory, and the driver's host allocation at CUDA init failed -- "No visible GPU devices", 2026-10-24)
    t0 = time.time()
    frames = rlhf_frames(args.rlhf, args.rlhf_n) + dataset_frames({"real": args.real, "aiet_grip": args.sim, "aiet_grip_lift": args.sim}, args.seed)
    print(f"{len(frames)} observations ({time.time() - t0:.0f} s)", flush=True)
    policy = policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
    model = policy._model
    graphdef, state = nnx.split(model)
    noise = np.asarray(jax.random.normal(jax.random.key(1234), (1, model.action_horizon, model.action_dim)), np.float32)

    @jax.jit
    def hidden(state, batch, noise):
        m = nnx.merge(graphdef, state)
        obs = _model.preprocess_observation(None, _model.Observation.from_dict(batch), train=False)
        prefix, prefix_mask, prefix_ar = m.embed_prefix(obs)
        b = prefix.shape[0]
        time_ = jnp.ones((b,))
        x_t = jnp.broadcast_to(noise, (b, *noise.shape[1:]))
        suffix, suffix_mask, suffix_ar, adarms = m.embed_suffix(obs, x_t, time_)
        mask = make_attn_mask(jnp.concatenate([prefix_mask, suffix_mask], 1), jnp.concatenate([prefix_ar, suffix_ar], 0))
        positions = jnp.cumsum(jnp.concatenate([prefix_mask, suffix_mask], 1), axis=1) - 1
        llm = m.PaliGemma.llm
        variables = bv.nnx_attrs_to_linen_vars({name: getattr(llm, name) for name in llm.linen_attributes})
        _, inter = llm.module.apply(variables, [prefix, suffix], mask=mask, positions=positions, adarms_cond=[None, adarms],
                                    mutable=["intermediates"])
        pre, suf = inter["intermediates"]["layers"]["hidden"][0]  # [layers, b, tokens, width]
        n_img = 256
        pooled = {cam: pre[:, :, k * n_img:(k + 1) * n_img].mean(2) for k, cam in enumerate(("env", "wrist", "overhead"))}
        text_mask = prefix_mask[:, 3 * n_img:].astype(jnp.float32)  # [b, text]
        pooled["text"] = (pre[:, :, 3 * n_img:] * text_mask[None, :, :, None]).sum(2) / text_mask.sum(1)[None, :, None]
        pooled["action"] = suf.mean(2)
        return {k: v.transpose(1, 0, 2) for k, v in pooled.items()}  # [b, layers, width]

    feats = {k: [] for k in ("env", "wrist", "overhead", "text", "action")}
    preds = []
    t0 = time.time()
    for b0 in range(0, len(frames), args.batch):
        items = frames[b0:b0 + args.batch]
        inputs = [policy._input_transform(jax.tree.map(lambda x: x, f["obs"])) for f in items]
        batch = jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *inputs)
        out = hidden(state, batch, noise)
        for k in feats:
            feats[k].append(np.asarray(out[k], np.float16))
        for f in items:  # (the checkpoint's own chunk for the voted decisions: the same noise for all)
            if f["source"] == "rlhf":
                preds.append(np.asarray(policy.infer(dict(f["obs"]), noise=noise[0])["actions"])[SKIP:SKIP + KEEP, :7])
        if (b0 // args.batch) % 20 == 0:
            print(f"  {b0 + len(items)}/{len(frames)} ({time.time() - t0:.0f} s)", flush=True)
    meta = {k: np.asarray([f.get(k, "") for f in frames]) for k in ("source", "kind", "label", "session", "base_note")}
    os.makedirs(OUT, exist_ok=True)
    out_path = f"{OUT}/acts_{args.name}.npz"
    np.savez(out_path, **{k: np.concatenate(v) for k, v in feats.items()}, **meta,
             episode=np.asarray([f["episode"] for f in frames]), progress=np.asarray([f["progress"] for f in frames], np.float32),
             rlhf_pred=np.asarray(preds, np.float32), rlhf_base=np.asarray([f["base"] for f in frames if f["source"] == "rlhf"], np.float32),
             config=args.config, checkpoint=args.checkpoint)
    print(f"{out_path}: {len(frames)} observations, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
