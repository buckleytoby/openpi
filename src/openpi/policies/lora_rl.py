"""(dexnex) LoRA reinforcement learning on a served PyTorch pi0.5: GRPO over the candidates the policy samples per
observation, rewarded by the RLHF critic, with a KL penalty to the base policy. Votes never reach this: they retrain the
critic (on the robot PC), which rewards the candidates.

Adapters: the served policy's action expert (the 300M "gemma_expert" transformer that denoises the action chunk) gets
low-rank adapters on its attention / MLP projections -- every chosen nn.Linear W becomes W + (alpha / rank) B A, A random,
B ZERO: at the start the policy is exactly the base policy. backbone: the language backbone (Gemma 2B, images + prompt)
too. Everything else frozen.

A decision = one observation and the N candidate chunks the policy sampled at it (one GRPO group), each rewarded by the
critic (2 P(up) - 1). Two kinds:
    live      the robot's decisions: the rlhf_residual node's batched infer (recorded here, "__rl_id" in the reply), then
              rl_score with the critic's rewards
    offline   frames of the served checkpoint's own training set (its LeRobot dataset on this machine, drawn with its
              training weights): rl_offline_draw samples N candidates at each with the current policy and returns what
              the critic needs (the candidates in action units, the frame's state and wrist image); the node scores them
              and sends rl_score back. They keep the policy's behavior on the training distribution in the updates.
              rlhf_frames_share of them come instead from past RLHF decisions' observations (rlhf_frames_dir: every
              earlier robot session's decision frames + state, extracted on the robot PC -- real robot states), re-sampled
              and re-scored the same way.

Replay buffer (buffer_size decisions): starts with offline_seed offline decisions -- and, when a LoRA is loaded (resume,
buffer_from_past), the newest past live decisions saved under rl_dir (their observations, candidates and critic rewards
then) --; each finished live decision replaces an offline one (the oldest live ones go once no offline one is left). An update (rl_update) is steps_per_update optimizer steps
(at least enough for every new live decision to be in one) of minibatch_records decisions each: at least
offline_fraction of them fresh offline decisions (drawn for this update: rl_offline_draw purpose "step"), then the new
live decisions, then random buffer entries (replays). Per decision:
    A_i        = (r_i - mean r) / max(std r, adv_std_floor)               (adv_std_norm false: r_i - mean r)
    L_i(theta) = candidate i's conditional flow-matching loss at mc_samples (tau, eps) draws fixed at the decision's first
                 update: || v_theta(x_tau, tau | obs) - (eps - a_i) ||^2, x_tau = tau eps + (1 - tau) a_i, summed over the
                 steps in window (the executed waypoints) and the first dims action dims (7: palm pose + gripper),
                 averaged over the draws -- only the likelihood estimate inside the ratio (no flow-matching target)
    ratio_i    = exp(ratio_scale * (L_i(theta_ref) - L_i(theta)))  (FPO; theta_ref = the policy at the decision's first
                 update: a replay keeps it, so the clip stops a stale decision from pushing far)
    pg_i       = -min(ratio_i A_i, clip(ratio_i, 1 - clip, 1 + clip) A_i)          (algorithm "fpo")
                 or w_i L_i(theta), w = exp(A_i / awr_beta) normalized to mean 1     (algorithm "awr")
    kl_i       = || v_theta(x_tau, tau) - v_base(x_tau, tau) ||^2 at the same draws and steps / dims (v_base: adapters off)
                 -- the KL between the two policies' denoising steps up to a constant: keeps the behavior near the base
    loss       = mean over the step's candidates of (pg_i + kl_coef kl_i)
AdamW on the LoRA weights, gradient-norm clip; the update stops early past target_kl (the ratio's mean (r - 1) - log r).
The prefix (images + prompt) runs once per decision without gradients (the frozen backbone's key / value cache), the
expert with them -- cheap next to serving (backbone true: the prefix keeps gradients too: slower).

Saved under rl_dir on this machine: <rl_dir>/<session>/lora_vNNN.pt per update, records/ (each decision: observation,
candidates, rewards), metrics.jsonl.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import numpy as np
import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """base(x) + (alpha / rank) x A^T B^T, A [rank, in] (Kaiming uniform), B [out, rank] (zeros: no change at the start).
    The adapter runs in float32 (the base stays bfloat16). ``weight`` / ``bias`` are the base's (gemma_pytorch reads
    ``o_proj.weight.dtype``)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank, self.scale = int(rank), float(alpha) / int(rank)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=dev, dtype=torch.float32))
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        d = (x.to(torch.float32) @ self.lora_A.t()) @ self.lora_B.t()
        return y + (d * self.scale).to(y.dtype)


def add_lora(model: nn.Module, rank: int, alpha: float, targets=DEFAULT_TARGETS, action_proj: bool = False,
             backbone: bool = False) -> dict[str, LoRALinear]:
    """LoRA on the action expert's attention / MLP projections named in ``targets`` (and its action in / out projections
    with ``action_proj``; and the same projections of the language backbone -- PaliGemma's Gemma 2B, which encodes the
    images + prompt -- with ``backbone``). Every other weight frozen (the vision encoder always). Returns {module path:
    LoRALinear}."""
    for p in model.parameters():
        p.requires_grad_(False)
    added = {}
    stacks = [("gemma_expert", model.paligemma_with_expert.gemma_expert.model.layers)]
    if backbone:
        stacks.append(("backbone", model.paligemma_with_expert.paligemma.language_model.layers))
    for stack, layers in stacks:
        for i, layer in enumerate(layers):
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name)
                for name in targets:
                    child = getattr(parent, name, None)
                    if isinstance(child, nn.Linear):
                        wrapped = LoRALinear(child, rank, alpha)
                        setattr(parent, name, wrapped)
                        added[f"{stack}.layers.{i}.{parent_name}.{name}"] = wrapped
    if action_proj:
        for name in ("action_in_proj", "action_out_proj"):
            wrapped = LoRALinear(getattr(model, name), rank, alpha)
            setattr(model, name, wrapped)
            added[name] = wrapped
    if not added:
        raise ValueError(f"LoRA: none of {targets} found in the action expert")
    return added


def remove_lora(model: nn.Module) -> None:
    """Unwrap every LoRALinear back to its base nn.Linear."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.base)




@dataclasses.dataclass
class RLConfig:
    # adapters (structural: changing them re-creates the LoRA, from zero)
    rank: int = 16
    alpha: float = 16.0
    targets: tuple = DEFAULT_TARGETS
    action_proj: bool = False
    backbone: bool = False  # also the language backbone (Gemma 2B: images + prompt) -- its pass then keeps gradients: slower
    # optimization
    algorithm: str = "fpo"  # fpo | awr
    lr: float = 3e-4
    weight_decay: float = 0.0
    steps_per_update: int = 8  # optimizer steps per update (at least enough for every new live decision to be in one)
    minibatch_records: int = 8  # decisions per optimizer step
    mc_samples: int = 4  # (tau, eps) draws per candidate for its likelihood estimate
    candidate_chunk: int = 16  # candidates per forward / backward pass (x mc_samples rows; GPU memory) -- the same gradient
    clip: float = 0.2
    ratio_scale: float = 1.0
    log_ratio_clamp: float = 5.0
    target_kl: float = 0.05
    kl_coef: float = 0.5  # the KL-to-base penalty's weight
    max_grad_norm: float = 1.0
    awr_beta: float = 0.5
    awr_max_weight: float = 20.0
    # advantages
    adv_std_norm: bool = True
    adv_std_floor: float = 0.03
    min_records: int = 4  # new live decisions needed for an update
    # replay buffer and offline decisions
    buffer_size: int = 256
    offline_seed: int = 0  # offline decisions the buffer starts with (the node draws and scores them at startup); 0 = none
    offline_fraction: float = 0.25  # at least this share of every optimizer step: fresh offline decisions
    # offline frames only from these sources (meta/episode_sources.json labels: real, aiet_grip, aiet_grip_lift, ...), with
    # the training weights among them -- the critic knows real wrist frames only; empty = every source, as trained
    offline_sources: tuple = ("real",)
    # past RLHF decisions' observations (avatar_intelligence rlhf_frames.py extract, copied here): this share of the offline
    # draws comes from them -- real robot states, re-sampled by the current policy and re-scored by the current critic
    rlhf_frames_dir: str = "~/rlhf_frames"
    rlhf_frames_share: float = 0.5
    # a LoRA loaded (resume): the replay buffer filled with the newest past live decisions (their saved records under
    # rl_dir: observation, candidates, critic rewards), up to buffer_size
    buffer_from_past: bool = True
    # bookkeeping
    max_pending: int = 256  # recorded batches waiting for their scores (the oldest dropped past this)
    save_records: bool = True
    save_every: int = 16  # lora_vNNN.pt (~110 MB with the optimizer state) every this many updates (and at rl_save)

    @classmethod
    def from_dict(cls, d: dict) -> "RLConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: (tuple(v) if k in ("targets", "offline_sources") else v) for k, v in d.items() if k in names}
        return cls(**kw)

    def structure(self) -> tuple:
        return (self.rank, float(self.alpha), tuple(self.targets), bool(self.action_proj), bool(self.backbone))

    def offline_per_step(self) -> int:
        return math.ceil(self.offline_fraction * self.minibatch_records) if self.offline_fraction > 0 else 0


@dataclasses.dataclass(eq=False)  # (compared by identity: "in the buffer")
class Record:
    rl_id: int
    obs: dict  # model-space inputs (numpy, after the input transforms), one observation
    actions: np.ndarray  # [N, H, D] normalized model-space chunks
    t: float
    version: int
    rewards: np.ndarray | None = None
    chosen: int = -1  # (live: the candidate that executed -- logged only)
    window: tuple = (0, 0)
    dims: int = 7
    t_scored: float = 0.0
    meta: dict = dataclasses.field(default_factory=dict)
    offline: str = ""  # "" live | "seed" (into the buffer) | "step" (one update's fresh offline share)
    # fixed at its first update: candidates with a nonzero advantage, their advantages, the (tau, eps) draws, their
    # flow-matching loss under the policy then (the ratio's reference) and the base policy's velocities there (the KL's)
    idx: np.ndarray | None = None
    adv: np.ndarray | None = None
    tau: torch.Tensor | None = None
    eps: torch.Tensor | None = None
    l_ref: torch.Tensor | None = None
    v_base: torch.Tensor | None = None
    replays: int = 0


def _np_tree(x):
    if isinstance(x, dict):
        return {k: _np_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_torch_obs(obs: dict, device) -> dict:
    """A record's numpy observation -> a fresh batch-of-1 torch tree (Observation.from_dict converts in place)."""
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        return torch.from_numpy(np.array(x)).to(device)[None, ...]
    return conv(obs)


def _uint8_hwc(img) -> np.ndarray:
    """A dataset image (uint8 HWC, or float CHW / HWC in [0, 1]) -> uint8 HWC."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
    return a


class OfflineSource:
    """Frames of the served checkpoint's own training set (its TrainConfig's LeRobot dataset, HF_LEROBOT_HOME), drawn
    with its source / view / cost weights when it has them, as the raw inputs the policy takes (after the dataset's repack
    transforms: "observation/..." keys + prompt -- what the inference node sends)."""

    def __init__(self, policy, sources=(), seed: int = 0, rlhf_dir: str = "", rlhf_share: float = 0.0):
        from openpi import transforms as _transforms
        from openpi.training import data_loader as _dl

        tc = getattr(policy, "train_config", None)
        if tc is None:
            raise ValueError("offline decisions: the served policy has no train_config (policy_config.create_trained_policy sets it)")
        data_config = tc.data.create(tc.assets_dirs, tc.model)
        self.repo_id = data_config.repo_id
        self.dataset = _dl.create_torch_dataset(data_config, tc.model.action_horizon, tc.model)
        self.repack = _transforms.compose(data_config.repack_transforms.inputs)
        self.n = len(self.dataset)
        w = (_dl.source_weighted_sampler(data_config, self.n, seed).weights.numpy().astype(np.float64)
             if data_config.source_weights is not None else np.ones(self.n))
        self.sources = tuple(sources or ())
        if self.sources:  # (frames of the other sources: weight 0)
            from lerobot.common.datasets import lerobot_dataset

            meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
            labels = json.loads((meta.root / "meta" / "episode_sources.json").read_text())
            lengths = [meta.episodes[i]["length"] for i in range(meta.total_episodes)]
            keep = np.concatenate([np.full(n, lab in self.sources) for n, lab in zip(lengths, labels)])
            if len(keep) != self.n or not (w * keep).any():
                raise ValueError(f"offline_sources {self.sources}: no frames of them in {self.repo_id} (its sources: {sorted(set(labels))})")
            w = w * keep
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)
        logging.info("LoRA RL offline decisions: frames of %s (%d of %d, %s; %s)", self.repo_id, int((self.p > 0).sum()), self.n,
                     f"sources {list(self.sources)}" if self.sources else "every source",
                     "its training weights" if data_config.source_weights is not None else "uniform")
        self.prompt = _np_tree(self.repack(self.dataset[int(np.flatnonzero(self.p)[0])])).get("prompt")
        self.rlhf = self._load_rlhf(rlhf_dir) if rlhf_dir and rlhf_share > 0 else None
        self.rlhf_share = float(rlhf_share) if self.rlhf else 0.0

    @staticmethod
    def _load_rlhf(rlhf_dir: str) -> dict | None:
        """Every shard of rlhf_frames.py (224 x 224 JPEGs + state per past decision), concatenated."""
        import glob

        files = sorted(glob.glob(os.path.join(os.path.expanduser(rlhf_dir), "*.npz")))
        if not files:
            logging.warning("LoRA RL: no past RLHF frames in %s (rlhf_frames.py extract, then copy here): training-set frames only", rlhf_dir)
            return None
        parts = {k: [] for k in ("keys", "wrist", "env", "overhead", "state")}
        for f in files:
            z = np.load(f, allow_pickle=True)
            for k in parts:
                parts[k].append(z[k])
        pool = {k: np.concatenate(v) for k, v in parts.items()}
        logging.info("LoRA RL offline decisions: + %d past RLHF decisions from %d sessions (%s)", len(pool["keys"]), len(files), rlhf_dir)
        return pool

    def _rlhf_raw(self, j: int) -> dict:
        """Past RLHF decision j as the inference node's raw inputs."""
        import io

        from PIL import Image

        img = lambda b: np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
        raw = {"observation/wrist_image": img(self.rlhf["wrist"][j]), "observation/env_image": img(self.rlhf["env"][j]),
               "observation/overhead_image": img(self.rlhf["overhead"][j]), "observation/state": self.rlhf["state"][j].astype(np.float32),
               "observation/env_image_valid": True}
        if self.prompt is not None:
            raw["prompt"] = self.prompt
        return raw

    def key(self, i: int) -> str:
        """A frame id's origin: "train:<dataset index>" or the past decision's "session:seq"."""
        return f"train:{i}" if i >= 0 else str(self.rlhf["keys"][-1 - i])

    def draw(self, k: int, indices=None) -> list[tuple[int, dict]]:
        """k frames -- rlhf_share of them past RLHF decisions (uniform), the rest the training set with its weights -- or
        these frame ids (>= 0: a dataset index; < 0: past RLHF decision -1 - id)."""
        if indices is None:
            n_rlhf = int(self.rng.binomial(k, self.rlhf_share)) if self.rlhf else 0
            indices = list(self.rng.choice(self.n, size=k - n_rlhf, p=self.p)) + \
                [-1 - int(j) for j in self.rng.integers(0, len(self.rlhf["keys"]), size=n_rlhf)] if self.rlhf else \
                list(self.rng.choice(self.n, size=k, p=self.p))
            self.rng.shuffle(indices)
        out = []
        for i in indices:
            i = int(i)
            if i < 0:
                out.append((i, self._rlhf_raw(-1 - i)))
                continue
            raw = _np_tree(self.repack(self.dataset[i]))
            raw.pop("actions", None)
            out.append((i, raw))
        return out


class LoRATrainer:
    """The adapters, the decisions and the update for one served PyTorch policy (openpi.policies.policy.Policy)."""

    def __init__(self, policy, config: RLConfig, rl_dir: str, session: str = ""):
        if not getattr(policy, "_is_pytorch_model", False):
            raise ValueError("LoRA RL needs a PyTorch checkpoint (this policy is JAX)")
        self.policy = policy
        self.model = policy._model
        self.device = policy._pytorch_device
        self.cfg = config
        self.session = session or time.strftime("%Y%m%d_%H%M%S")
        self.dir = pathlib.Path(rl_dir).expanduser() / self.session
        self.dir.mkdir(parents=True, exist_ok=True)
        self.modules = add_lora(self.model, config.rank, config.alpha, config.targets, config.action_proj, config.backbone)
        self.params = [p for m in self.modules.values() for p in (m.lora_A, m.lora_B)]
        self.opt = torch.optim.AdamW(self.params, lr=config.lr, weight_decay=config.weight_decay)
        self.records: dict[int, Record] = {}  # (recorded, waiting for their scores)
        self.buffer: list[Record] = []  # (the replay buffer, oldest first)
        self.new_queue: list[Record] = []  # (live decisions in the buffer not yet in an optimizer step)
        self.offline_queue: list[Record] = []  # (fresh offline decisions for the next update's steps)
        self.offline = None
        self.next_id = 0
        self.version = 0
        self.enabled = True
        self.last_saved = ""
        n = sum(p.numel() for p in self.params)
        logging.info("LoRA RL: %d adapters, %.2fM trainable parameters (rank %d, alpha %g); saving to %s",
                     len(self.modules), n / 1e6, config.rank, config.alpha, self.dir)
        self._reset_compiled()
        policy.rl_recorder = self.record

    # ── adapters ─────────────────────────────────────────────────────────────
    def _reset_compiled(self):
        """The served sample function compiled before the adapters existed: dropped (recompiled on its next use)."""
        torch._dynamo.reset()
        if getattr(self.policy, "_sample_compiled", None) is not None:
            self.policy._sample_compiled = None
        self.policy._sample_actions = self.policy._sample_eager

    def detach(self):
        self.policy.rl_recorder = None
        remove_lora(self.model)
        self._reset_compiled()

    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        for m in self.modules.values():
            m.enabled = self.enabled

    def reconfigure(self, config: RLConfig):
        """Non-structural settings (lr, clip, buffer, ...) changed in place; the optimizer's lr / weight decay too."""
        self.cfg = config
        for g in self.opt.param_groups:
            g["lr"], g["weight_decay"] = config.lr, config.weight_decay
        self._evict()

    def _evict(self):
        """Past buffer_size: offline decisions go first (live ones replace them), then the oldest live ones."""
        while len(self.buffer) > self.cfg.buffer_size:
            off = next((i for i, r in enumerate(self.buffer) if r.offline), None)
            gone = self.buffer.pop(off if off is not None else 0)
            if gone in self.new_queue:
                self.new_queue.remove(gone)

    def reset(self):
        """B back to zero (the base policy again), A redrawn, a fresh optimizer."""
        with torch.no_grad():
            for m in self.modules.values():
                nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
                m.lora_B.zero_()
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.version = 0

    def state_dict(self) -> dict:
        return {"lora": {k: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()} for k, m in self.modules.items()},
                "structure": self.cfg.structure(), "config": dataclasses.asdict(self.cfg), "version": self.version,
                "optimizer": self.opt.state_dict()}

    def save(self, path: str | None = None) -> str:
        path = str(path or self.dir / f"lora_v{self.version:03d}.pt")
        torch.save(self.state_dict(), path)
        self.last_saved = path
        return path

    def load(self, path: str) -> int:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        st = tuple(sd["structure"]) + ((False,) if len(sd["structure"]) < 5 else ())
        if (st[0], tuple(st[2]), bool(st[3]), bool(st[4])) != (self.cfg.rank, tuple(self.cfg.targets), bool(self.cfg.action_proj),
                                                               bool(self.cfg.backbone)):
            raise ValueError(f"{path}: adapters {sd['structure']} differ from the configured {self.cfg.structure()}")
        with torch.no_grad():
            for k, m in self.modules.items():
                m.lora_A.copy_(sd["lora"][k]["A"].to(m.lora_A.device))
                m.lora_B.copy_(sd["lora"][k]["B"].to(m.lora_B.device))
                m.scale = float(st[1]) / m.rank
        try:
            self.opt.load_state_dict(sd["optimizer"])
        except Exception as e:  # noqa: BLE001
            logging.warning("LoRA RL: %s: optimizer state not loaded (%s) -- a fresh one", path, e)
        self.version = int(sd.get("version", 0))
        return self.version

    def lora_norm(self) -> float:
        """|| sum of the scaled B A updates || over all adapters (Frobenius): how far the policy has moved (0 = base)."""
        with torch.no_grad():
            return float(math.sqrt(sum(float(((m.lora_B @ m.lora_A) * m.scale).pow(2).sum()) for m in self.modules.values())))

    # ── decisions ────────────────────────────────────────────────────────────
    def record(self, inputs_np: dict, actions_model: np.ndarray, num_samples: int) -> int | None:
        """Policy.infer's hook: a batched call (num_samples > 1) recorded. Returns its rl_id."""
        if num_samples < 2:
            return None
        rl_id = self.next_id
        self.next_id += 1
        self.records[rl_id] = Record(rl_id, inputs_np, np.asarray(actions_model, np.float32).reshape(num_samples, *np.shape(actions_model)[-2:]),
                                     time.time(), self.version)
        pending = [r for r in self.records.values() if r.rewards is None]
        while len(pending) > self.cfg.max_pending:  # (never scored: e.g. a batch the node didn't use)
            self.records.pop(pending.pop(0).rl_id, None)
        return rl_id

    def load_past_records(self) -> int:
        """The replay buffer filled with the newest past live decisions saved under rl_dir (every session's records/: the
        observation, the candidates and their critic rewards then), up to buffer_size. Their ratio reference is set at
        their first update here, under the current policy. Returns how many."""
        import glob

        files = sorted(glob.glob(str(self.dir.parent / "*" / "records" / "rec_*.pt")), key=os.path.getmtime, reverse=True)
        past = []
        for f in files:
            if len(past) + len(self.buffer) >= self.cfg.buffer_size:
                break
            try:
                d = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001
                continue
            if d.get("offline") or d.get("rewards") is None:  # (live decisions only)
                continue
            rec = Record(self.next_id, d["obs"], np.asarray(d["actions"], np.float32), d.get("t", 0.0), d.get("version", -1),
                         rewards=np.asarray(d["rewards"], np.float32), chosen=int(d.get("chosen", -1)), window=tuple(d.get("window", (0, 0))),
                         dims=int(d.get("dims", 7)), meta={**(d.get("meta") or {}), "past": os.path.relpath(f, self.dir.parent)})
            self.next_id += 1
            if np.any(np.abs(self.advantages(rec)) > 1e-6):
                past.append(rec)
        self.buffer = past[::-1] + self.buffer  # (oldest first)
        self._evict()
        logging.info("LoRA RL: replay buffer filled with %d past live decisions (of %d saved records)", len(past), len(files))
        return len(past)

    def _resolve(self, rl_id) -> Record:
        if rl_id in (None, "last", -1):
            if not self.records:
                raise KeyError("no recorded batch yet")
            return self.records[max(self.records)]
        rec = self.records.get(int(rl_id))
        if rec is None:
            raise KeyError(f"rl_id {rl_id}: no such record (expired, or recorded before a restart)")
        return rec

    def offline_draw(self, k: int, num_samples: int, noise_temperature, purpose: str, indices=None) -> dict:
        """k training-set frames (or these dataset indices), num_samples candidates sampled at each by the current policy
        (recorded as offline decisions, purpose "seed" or "step"; "eval": not recorded -- e.g. a fixed set re-scored after
        every update): per frame its rl_id, the candidates in action units [N, H, d], the state and the wrist image (what
        the critic needs)."""
        if self.offline is None:
            self.offline = OfflineSource(self.policy, self.cfg.offline_sources, rlhf_dir=self.cfg.rlhf_frames_dir,
                                         rlhf_share=self.cfg.rlhf_frames_share)
        frames = []
        for i, raw in self.offline.draw(int(k), None if indices is None else [int(j) for j in indices]):
            obs = dict(raw)
            obs["__num_samples"] = int(num_samples)
            obs["__noise_temperature"] = np.asarray(noise_temperature, np.float32)
            obs["__compile"] = False
            out = self.policy.infer(obs)
            rl_id = out.get("__rl_id")
            if rl_id is None:
                raise RuntimeError("offline draw: the batch wasn't recorded")
            if purpose == "eval":
                self.records.pop(rl_id, None)
            else:
                rec = self.records[rl_id]
                rec.offline, rec.meta = purpose, {"frame": i, "origin": self.offline.key(i)}
            wrist = next((raw[k_] for k_ in ("observation/wrist_image", "wrist_image") if k_ in raw), None)
            state = next((raw[k_] for k_ in ("observation/state", "state") if k_ in raw), None)
            frames.append({"rl_id": int(rl_id), "frame": i, "actions": np.asarray(out["actions"], np.float32),
                           "state": None if state is None else np.asarray(state, np.float32),
                           "wrist": None if wrist is None else _uint8_hwc(wrist)})
        return {"success": True, "frames": frames, "repo_id": self.offline.repo_id}

    def score(self, rl_id, rewards, chosen: int = -1, window=(0, 0), dims: int = 7, meta: dict | None = None) -> int:
        """The critic's rewards for a recorded decision's candidates: it's finished -- into the buffer (live, offline
        "seed"), or the next update's offline share ("step")."""
        rec = self._resolve(rl_id)
        rewards = np.asarray(rewards, np.float32).reshape(-1)
        if len(rewards) != len(rec.actions):
            raise ValueError(f"rl_id {rec.rl_id}: {len(rewards)} rewards for {len(rec.actions)} candidates")
        rec.rewards, rec.chosen, rec.dims, rec.t_scored = rewards, int(chosen), int(dims), time.time()
        h = rec.actions.shape[1]
        s0, s1 = int(window[0]), int(window[1])
        rec.window = (max(0, s0), min(h, s1)) if s1 > s0 else (0, h)
        rec.meta.update(meta or {})
        self.records.pop(rec.rl_id, None)
        if self.cfg.save_records:
            rec_dir = self.dir / "records"
            rec_dir.mkdir(exist_ok=True)
            keep = ("rl_id", "obs", "actions", "t", "version", "rewards", "chosen", "window", "dims", "meta", "offline")
            torch.save({k: getattr(rec, k) for k in keep} | {"advantages": self.advantages(rec)}, rec_dir / f"rec_{rec.rl_id:06d}.pt")
        if not np.any(np.abs(self.advantages(rec)) > 1e-6):  # (all rewarded alike: nothing to learn from it)
            return rec.rl_id
        if rec.offline == "step":
            self.offline_queue.append(rec)
        else:
            self.buffer.append(rec)
            if not rec.offline:
                self.new_queue.append(rec)
            self._evict()
        return rec.rl_id

    def advantages(self, rec: Record) -> np.ndarray:
        r = rec.rewards.astype(np.float64)
        a = r - r.mean()
        if self.cfg.adv_std_norm:
            a = a / max(float(r.std()), self.cfg.adv_std_floor)
        return a.astype(np.float32)

    # ── the candidates' velocities ───────────────────────────────────────────
    def _prefix(self, rec: Record):
        """The decision's observation through the backbone: its key / value cache (per layer (k, v), batch 1), pad mask and
        state. Without gradients (frozen) -- unless the backbone has adapters and gradients are on."""
        from openpi.models import model as _model
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        m = self.model
        obs = _model.Observation.from_dict(_to_torch_obs(rec.obs, self.device))
        with torch.set_grad_enabled(self.cfg.backbone and torch.is_grad_enabled()):
            images, img_masks, lang, lang_masks, state = m._preprocess_observation(obs, train=False)
            embs, pad, att = m.embed_prefix(images, img_masks, lang, lang_masks)
            att_4d = m._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
            m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, pkv = m.paligemma_with_expert.forward(attention_mask=att_4d, position_ids=torch.cumsum(pad, dim=1) - 1,
                                                     past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
        return [(pkv[i][0], pkv[i][1]) for i in range(len(pkv))], pad, state

    def _velocities(self, rec: Record, rows: np.ndarray, prefix=None):
        """Candidates rec.idx[rows] at the decision's fixed draws: (L [len(rows), M] -- each one's flow-matching loss over
        window x dims --, v [len(rows) M, w, dims] -- the velocities there), with gradients when enabled."""
        layers, pad, state = prefix if prefix is not None else self._prefix(rec)
        n, m_ = len(rows), rec.tau.shape[1]
        c = n * m_
        from transformers.cache_utils import DynamicCache

        pkv = DynamicCache.from_legacy_cache(tuple((k.repeat_interleave(c, 0), v.repeat_interleave(c, 0)) for k, v in layers))
        a0 = torch.from_numpy(rec.actions[rec.idx[rows]]).to(self.device).repeat_interleave(m_, 0)  # [n M, H, D]
        t = rec.tau[rows].reshape(-1).to(self.device)
        e = rec.eps[rows].reshape(-1, *rec.eps.shape[2:]).to(self.device)
        x_t = t[:, None, None] * e + (1 - t[:, None, None]) * a0
        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        s0, s1 = rec.window
        v = self.model.denoise_step(state.repeat_interleave(c, 0), pad.repeat_interleave(c, 0), pkv, x_t, t).float()[:, s0:s1, : rec.dims]
        u = (e - a0)[:, s0:s1, : rec.dims]
        return (v - u).pow(2).sum(dim=(1, 2)).view(n, m_), v

    def _chunks(self, rec: Record):
        k = max(1, self.cfg.candidate_chunk)
        return [np.arange(i, min(i + k, len(rec.idx))) for i in range(0, len(rec.idx), k)]

    def _prepare(self, rec: Record):
        """A decision's first update: its candidates with a nonzero advantage, the fixed (tau, eps) draws, their loss
        under the current policy (the ratio's reference from now on) and the base policy's velocities (adapters off)."""
        if rec.l_ref is not None:
            return
        adv = self.advantages(rec)
        rec.idx = np.flatnonzero(np.abs(adv) > 1e-6)
        rec.adv = adv[rec.idx]
        h, d = rec.actions.shape[1:]
        rec.tau = torch.distributions.Beta(1.5, 1.0).sample((len(rec.idx), self.cfg.mc_samples)) * 0.999 + 0.001
        rec.eps = torch.randn((len(rec.idx), self.cfg.mc_samples, h, d))
        with torch.no_grad():
            prefix = self._prefix(rec)
            rec.l_ref = torch.cat([self._velocities(rec, rows, prefix)[0].mean(1) for rows in self._chunks(rec)]).detach()
            self.set_enabled(False)
            try:
                base_prefix = self._prefix(rec) if self.cfg.backbone else prefix
                rec.v_base = torch.cat([self._velocities(rec, rows, base_prefix)[1] for rows in self._chunks(rec)]).detach()
            finally:
                self.set_enabled(True)

    def _decision_backward(self, rec: Record, n_total: int, stats: dict):
        """The decision's loss (its share of the step's mean), back-propagated chunk by chunk (gradients accumulate)."""
        cfg = self.cfg
        prefix = None if cfg.backbone else self._prefix(rec)  # (frozen backbone: one prefix for every chunk)
        m_ = cfg.mc_samples
        logs = {"kl": [], "ratio": [], "log_ratio": [], "pg": 0.0, "klb": []}
        for rows in self._chunks(rec):
            l_new = self._velocities(rec, rows, prefix)
            l_new, v = l_new[0].mean(1), l_new[1]
            adv = torch.from_numpy(rec.adv[rows]).to(l_new.device)
            l_ref = rec.l_ref[rows]
            if cfg.algorithm == "awr":
                w = torch.exp(torch.from_numpy(rec.adv).to(l_new.device) / cfg.awr_beta).clamp(max=cfg.awr_max_weight)
                w = w[rows] / w.mean()
                pg = (w * l_new).sum()
                log_ratio = (l_ref - l_new).detach() * cfg.ratio_scale
                ratio = log_ratio.exp()
            else:
                log_ratio = ((l_ref - l_new) * cfg.ratio_scale).clamp(-cfg.log_ratio_clamp, cfg.log_ratio_clamp)
                ratio = log_ratio.exp()
                pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).sum()
            vb = rec.v_base[rows[0] * m_:(rows[-1] + 1) * m_]
            kl = (v - vb).pow(2).sum(dim=(1, 2)).view(len(rows), m_).mean(1)
            ((pg + cfg.kl_coef * kl.sum()) / n_total).backward()
            with torch.no_grad():
                logs["ratio"].append(ratio.detach())
                logs["log_ratio"].append(log_ratio.detach())
                logs["pg"] += float(pg)
                logs["klb"].append(kl.detach())
        ratio, log_ratio = torch.cat(logs["ratio"]), torch.cat(logs["log_ratio"])
        stats["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
        stats["ratio_mean"].append(float(ratio.mean()))
        stats["clip_frac"].append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
        stats["pg_loss"].append(logs["pg"] / max(len(rec.idx), 1))
        stats["kl_base"].append(float(torch.cat(logs["klb"]).mean()))
        stats["offline" if rec.offline else ("replay" if rec.replays else "fresh")].append(1.0)
        rec.replays += 1

    # ── the update ───────────────────────────────────────────────────────────
    def update(self, offline_only: bool = False) -> dict:
        """Optimizer steps over the replay buffer: each step at least offline_fraction fresh offline decisions, then the
        live decisions not yet trained on, then random buffer entries (replays, the seeded offline decisions).
        offline_only (no robot -- rlhf_lora_offline.py): every step's slots fresh offline decisions (the buffer's
        replays fill in when they run short); no live decisions needed."""
        cfg = self.cfg
        if offline_only:
            if not self.offline_queue:
                return {"success": True, "updated": False, "ready": 0,
                        "message": "offline_only: no fresh offline decisions (rl_offline_draw purpose 'step', then rl_score)"}
        elif len(self.new_queue) < cfg.min_records:
            return {"success": True, "updated": False, "ready": len(self.new_queue),
                    "message": f"{len(self.new_queue)} new live decisions (< min_records {cfg.min_records})"}
        t0 = time.time()
        queue, off_queue = ([] if offline_only else list(self.new_queue)), list(self.offline_queue)
        self.offline_queue = []  # (fresh for this update only)
        for rec in queue + off_queue:  # (their references: the policy before this update's steps)
            self._prepare(rec)
        n_off = cfg.minibatch_records if offline_only else cfg.offline_per_step()
        k_live = 0 if offline_only else max(1, cfg.minibatch_records - n_off)
        steps = cfg.steps_per_update if offline_only else max(cfg.steps_per_update, math.ceil(len(queue) / k_live))
        np.random.shuffle(queue)
        if n_off and len(off_queue) < n_off * steps and not offline_only:
            logging.warning("LoRA RL: %d fresh offline decisions for %d steps x %d: the rest of the steps without", len(off_queue), steps, n_off)
        stats = {k: [] for k in ("pg_loss", "kl_base", "ratio_mean", "clip_frac", "approx_kl", "grad_norm", "fresh", "replay", "offline")}
        steps_run, early = 0, False
        for _ in range(steps):
            batch = [off_queue.pop() for _ in range(min(n_off, len(off_queue)))]
            fresh = [queue.pop() for _ in range(min(k_live, len(queue)))]
            pool = [r for r in self.buffer if r not in fresh]
            n_extra = min(cfg.minibatch_records - len(batch) - len(fresh), len(pool))
            extra = [pool[i] for i in np.random.choice(len(pool), size=n_extra, replace=False)] if n_extra > 0 else []
            for rec in extra:
                self._prepare(rec)
            batch += fresh + extra
            n_total = sum(len(r.idx) for r in batch)
            if n_total == 0:
                continue
            kl_before = len(stats["approx_kl"])
            self.opt.zero_grad(set_to_none=True)
            for rec in batch:  # (one decision at a time, in candidate chunks: gradients accumulate)
                self._decision_backward(rec, n_total, stats)
            stats["grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)))
            self.opt.step()
            steps_run += 1
            for rec in fresh:
                if rec in self.new_queue:
                    self.new_queue.remove(rec)
            kl_step = stats["approx_kl"][kl_before:]
            if kl_step and float(np.mean(kl_step)) > cfg.target_kl:
                early = True  # (live decisions left in the queue: the next update's)
                break
        self.opt.zero_grad(set_to_none=True)
        self.version += 1
        path = self.save() if self.version % max(1, cfg.save_every) == 0 or not self.last_saved else self.last_saved
        live = [r for r in self.buffer if not r.offline]
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        metrics = {
            "version": self.version, "steps": steps_run, "early_stop": early,
            "records": len(stats["fresh"]), "replayed": len(stats["replay"]), "offline_used": len(stats["offline"]),
            "buffer_decisions": len(live), "buffer_offline": len(self.buffer) - len(live), "queue_left": len(self.new_queue),
            "pg_loss": mean(stats["pg_loss"]), "kl_base": mean(stats["kl_base"]), "ratio_mean": mean(stats["ratio_mean"]),
            "clip_frac": mean(stats["clip_frac"]), "approx_kl": mean(stats["approx_kl"]), "grad_norm": mean(stats["grad_norm"]),
            "reward_std_within": mean([float(r.rewards.std()) for r in live]),
            "lora_norm": self.lora_norm(), "seconds": time.time() - t0, "saved": path,
        }
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), **metrics}) + "\n")
        logging.info("LoRA RL update v%d: %d steps, %d new + %d replayed live + %d offline decisions, %.1f s | pg %.4f kl_base %.4f "
                     "ratio %.3f clip %.2f kl %.4f | buffer %d live + %d offline | |BA| %.4f", self.version, steps_run,
                     metrics["records"], metrics["replayed"], metrics["offline_used"], metrics["seconds"], metrics["pg_loss"],
                     metrics["kl_base"], metrics["ratio_mean"], metrics["clip_frac"], metrics["approx_kl"],
                     metrics["buffer_decisions"], metrics["buffer_offline"], metrics["lora_norm"])
        return {"success": True, "updated": True, **metrics}

    def status(self) -> dict:
        return {"success": True, "version": self.version, "enabled": self.enabled, "dir": str(self.dir),
                "pending": len(self.records), "buffer_decisions": sum(not r.offline for r in self.buffer),
                "buffer_offline": sum(bool(r.offline) for r in self.buffer), "new_queue": len(self.new_queue),
                "offline_queue": len(self.offline_queue), "offline_set": getattr(self.offline, "repo_id", None),
                "lora_norm": self.lora_norm(), "last_saved": self.last_saved, "config": dataclasses.asdict(self.cfg)}


class RLControl:
    """The server's "rl_*" control commands (serve_policy.py). The trainer is attached to the served policy at
    rl_configure, and again whenever the served policy object changed (a reload: the adapters start from zero unless
    the configure carries a resume path)."""

    def __init__(self, get_policy, rl_dir: str):
        self.get_policy = get_policy
        self.rl_dir = rl_dir
        self.trainer: LoRATrainer | None = None

    def __call__(self, command: dict) -> dict:
        cmd = command.get("cmd")
        try:
            if cmd == "rl_configure":
                return self._configure(command)
            policy = self.get_policy()
            if self.trainer is None or self.trainer.policy is not policy:
                return {"success": False, "code": "not_configured",
                        "message": "LoRA RL not configured for the served policy (rl_configure first; a reload needs it again)"}
            t = self.trainer
            if cmd == "rl_score":
                rl_id = t.score(command.get("rl_id"), command["rewards"], int(command.get("chosen", -1)),
                                tuple(command.get("window", (0, 0))), int(command.get("dims", 7)), command.get("meta"))
                return {"success": True, "rl_id": rl_id}
            if cmd == "rl_offline_draw":
                return t.offline_draw(int(command.get("k", 1)), int(command["num_samples"]),
                                      command.get("noise_temperature", 1.0), str(command.get("purpose", "step")),
                                      command.get("indices"))
            if cmd == "rl_update":
                return t.update(offline_only=bool(command.get("offline_only", False)))
            if cmd == "rl_status":
                return t.status()
            if cmd == "rl_save":
                return {"success": True, "path": t.save(command.get("path"))}
            if cmd == "rl_load":
                return {"success": True, "version": t.load(command["path"])}
            if cmd == "rl_reset":
                t.reset()
                return {"success": True, "version": 0}
            if cmd == "rl_enable":
                t.set_enabled(bool(command.get("on", True)))
                return {"success": True, "enabled": t.enabled}
            return {"success": False, "message": f"unknown RL command {cmd!r}"}
        except Exception as e:  # noqa: BLE001
            logging.exception("RL command %s failed", cmd)
            return {"success": False, "message": f"{type(e).__name__}: {e}"}

    def _configure(self, command: dict) -> dict:
        policy = self.get_policy()
        if policy is None:
            return {"success": False, "message": "no policy served yet"}
        cfg = RLConfig.from_dict(command.get("config", {}))
        t = self.trainer
        fresh = False
        session = command.get("session", "")
        same_session = t is not None and (not session or session == t.session)  # (another session: its own LoRA, from zero)
        if t is not None and t.policy is policy and t.cfg.structure() == cfg.structure() and same_session:
            t.reconfigure(cfg)
        else:
            if t is not None and t.policy is policy:
                t.detach()
            self.trainer = t = LoRATrainer(policy, cfg, self.rl_dir, session)
            fresh = True
        resume = command.get("resume") or ""
        if resume and t.version == 0:
            if not os.path.exists(resume):
                return {"success": False, "message": f"resume {resume}: not found on the server ({os.uname().nodename})"}
            t.load(resume)
            if cfg.buffer_from_past:
                t.load_past_records()
        t.set_enabled(bool(command.get("enabled", True)))
        return {"success": True, "version": t.version, "dir": str(t.dir), "host": os.uname().nodename, "fresh": fresh,
                "buffer_decisions": sum(not r.offline for r in t.buffer),
                "buffer_offline": sum(bool(r.offline) for r in t.buffer),
                "trainable": int(sum(p.numel() for p in t.params)), "adapters": len(t.modules)}
