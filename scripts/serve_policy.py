import asyncio
import dataclasses
import enum
import logging
import os
import pathlib
import socket
import sys

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# Sentinel selection meaning "serve the raw, un-fine-tuned base checkpoint"
# (see BASE_CHECKPOINT_DIR / _ReloadHandler) -- distinct from any real
# step-directory name a --checkpoint-root scan could ever produce.
BASE_CHECKPOINT_SELECTION = "base"

# The raw pi0.5 base checkpoint (pre-fine-tuning) -- same path
# pi05_dexnex_erlenmeyer's weight_loader fine-tunes FROM (see config.py),
# and the same directory layout (params/ + assets/) as any release
# checkpoint / anything train.py's own checkpoint_manager writes, so
# _policy_config.create_trained_policy can load it exactly the same way as
# a fine-tuned checkpoint under --checkpoint-root.
BASE_CHECKPOINT_DIR = "gs://openpi-assets/checkpoints/pi05_base"


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # If set, enables hot-reloadable checkpoint selection via control messages
    # (see websocket_policy_server.CONTROL_KEY / _ReloadHandler below) -- a
    # client can list checkpoint subdirectories under this root (each expected
    # to contain a "params" subfolder, i.e. the same layout train.py's
    # checkpoint_manager writes) and switch the served policy between any of
    # them, or the raw un-fine-tuned base checkpoint, without restarting this
    # process. Only supported when `policy` is a Checkpoint (need a fixed
    # train_config to rebuild the model/transforms against on every reload).
    # e.g. "checkpoints/pi05_dexnex_erlenmeyer/run1_h100_12h"
    checkpoint_root: str | None = None

    # (dexnex) If set, hot-reload across MODEL FAMILIES instead: a JSON file listing them --
    #   {"default": "<key>/<step>", "models": [{"key", "root", "config", "description"}, ...]}
    # -- "root" a folder of training-step checkpoints (each with params/: JAX, or model.safetensors: PyTorch), "config" the training config to build the
    # model / transforms with. The selections are "<key>/<step>", each loaded with ITS family's config and its own
    # checkpoint's norm stats (its assets/ -- the families' data differ, so one fixed set can't serve them all). The
    # client gets the descriptions with the list (the inference node's GUI shows them). Replaces --checkpoint-root and
    # --policy (the "default" selection is served at startup).
    model_registry: str | None = None

    # (dexnex) If set, LoRA reinforcement learning on the served PyTorch policy (openpi/policies/lora_rl.py): the
    # "rl_*" control commands (the rlhf_residual node's lora_rl mode drives them). Adapters, records and metrics are
    # saved under this folder on THIS machine, e.g. ~/rlhf_lora.
    rl_dir: str | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


class _ReloadHandler:
    """Handles hot-reload control commands (see Args.checkpoint_root):
        {"cmd": "list_checkpoints"}                    -> {"success", "checkpoints": [...], "current": ...}
        {"cmd": "load_checkpoint", "selection": ...}    -> {"success", "message", "current": ...}
        {"cmd": "current_checkpoint"}                   -> {"success", "current": ...}

    "selection" is either BASE_CHECKPOINT_SELECTION ("base", the raw
    un-fine-tuned checkpoint) or the name of a subdirectory of
    checkpoint_root (a training step, e.g. "20000"). Every rebuilt policy
    always uses the SAME fixed norm_stats (computed once from train_config's
    own assets, i.e. whatever compute_norm_stats.py wrote for this config) --
    deliberately NOT each checkpoint's own bundled assets/ dir, so switching
    between base and fine-tuned checkpoints is an apples-to-apples
    comparison of WEIGHTS only, never silently also comparing different
    normalization stats. The base checkpoint's own bundled assets (generic
    pi0.5 pretraining norm stats) would otherwise misinterpret this robot's
    state/action units entirely.
    """

    def __init__(
        self,
        server: websocket_policy_server.WebsocketPolicyServer,
        train_config: _config.TrainConfig,
        checkpoint_root: pathlib.Path,
        default_prompt: str | None,
        registry: dict | None = None,
    ):
        self._server = server
        self._train_config = train_config
        self._checkpoint_root = checkpoint_root
        self._default_prompt = default_prompt
        self._current = BASE_CHECKPOINT_SELECTION
        self._registry = registry  # (--model-registry: families of checkpoints, each with its own config)
        if registry is not None:
            self._norm_stats = None
            return

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        self._norm_stats = data_config.norm_stats
        if self._norm_stats is None:
            logging.warning(
                "No norm stats found for %s -- run compute_norm_stats.py first. Reload will fail "
                "until then.",
                train_config.name,
            )

    def _registry_selections(self) -> dict[str, tuple[dict, pathlib.Path]]:
        """--model-registry: every "<key>/<step>" -> (its family entry, its checkpoint folder)."""
        out = {}
        for entry in self._registry["models"]:
            root = pathlib.Path(entry["root"])
            if not root.exists():
                continue
            steps = sorted((p for p in root.iterdir() if p.is_dir() and ((p / "params").exists() or (p / "model.safetensors").exists())),
                           key=lambda p: int(p.name) if p.name.isdigit() else p.name)
            for step in steps:
                out[f"{entry['key']}/{step.name}"] = (entry, step)
        return out

    def _list_checkpoint_dirs(self) -> list[str]:
        if not self._checkpoint_root.exists():
            return []
        return sorted(
            (p.name for p in self._checkpoint_root.iterdir() if p.is_dir() and (p / "params").exists()),
            key=lambda name: int(name) if name.isdigit() else name,
        )

    def _build_policy(self, selection: str, compile: bool = False) -> _policy.Policy:
        if self._registry is not None:  # (the family's own config; norm stats from the checkpoint's own assets/)
            entry, checkpoint_dir = self._registry_selections()[selection]
            train_config = _config.get_config(entry["config"])
            if hasattr(train_config.model, "pytorch_compile_mode"):  # (PyTorch: off -- quick tests -- or SERVE_COMPILE_MODE; JAX unaffected)
                mode = _policy.SERVE_COMPILE_MODE if compile else None
                train_config = dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=mode))
            return _policy_config.create_trained_policy(train_config, checkpoint_dir, default_prompt=self._default_prompt)
        if selection == BASE_CHECKPOINT_SELECTION:
            checkpoint_dir = BASE_CHECKPOINT_DIR
        else:
            checkpoint_dir = self._checkpoint_root / selection
        return _policy_config.create_trained_policy(
            self._train_config,
            checkpoint_dir,
            default_prompt=self._default_prompt,
            norm_stats=self._norm_stats,
        )

    def __call__(self, command: dict) -> dict:
        cmd = command.get("cmd")
        try:
            if cmd == "list_checkpoints":
                if self._registry is not None:
                    selections = self._registry_selections()
                    descriptions = {sel: f"{entry['description']}  [config {entry['config']}; {path}]"
                                    for sel, (entry, path) in selections.items()}
                    return {"success": True, "checkpoints": list(selections), "descriptions": descriptions, "current": self._current}
                checkpoints = [BASE_CHECKPOINT_SELECTION, *self._list_checkpoint_dirs()]
                return {"success": True, "checkpoints": checkpoints, "current": self._current}
            if cmd == "current_checkpoint":
                return {"success": True, "current": self._current}
            if cmd == "load_checkpoint":
                selection = command.get("selection")
                known = (self._registry_selections() if self._registry is not None
                         else [BASE_CHECKPOINT_SELECTION, *self._list_checkpoint_dirs()])
                if selection not in known:
                    return {"success": False, "message": f"Unknown checkpoint selection: {selection!r}"}
                compile_ = bool(command.get("compile", False))
                if self._registry is not None:  # (a fresh process: see _restart_with)
                    logging.info("load_checkpoint %s (PyTorch torch.compile: %s): restarting the server with it", selection,
                                 "on" if compile_ else "off")
                    asyncio.get_running_loop().call_later(0.2, _restart_with, selection, compile_)  # (after this reply is sent)
                    return {"success": True, "message": f"Restarting the server with {selection}", "current": selection,
                            "restarting": True}
                # Drop the OLD policy's GPU-resident weights BEFORE building
                # the new one -- otherwise _build_policy loads the new
                # weights onto the GPU while the old policy (still
                # referenced by self._server until set_policy below) is
                # still resident too, briefly needing ~2x a model's memory.
                # Fine on an 80GB datacenter GPU, but can OOM on a
                # consumer-sized one (e.g. a 24GB RTX 3090) even for a
                # single pi0.5-sized model. gc.collect() is needed (not just
                # `del`) since jax.Array objects can be kept alive by
                # reference cycles the refcounting alone won't break.
                self._server.set_policy(None)
                _release_gpu_memory()
                logging.info("load_checkpoint %s (PyTorch torch.compile: %s; request %s)", selection, "on" if compile_ else "off", command)
                policy = self._build_policy(selection, compile=compile_)
                self._server.set_policy(policy, metadata=policy.metadata)
                self._current = selection
                return {"success": True, "message": f"Loaded {selection}", "current": self._current}
            return {"success": False, "message": f"Unknown control command: {cmd!r}"}
        except Exception as e:  # noqa: BLE001 -- report any load/build failure back to the client, don't crash the server
            logging.exception("Control command failed")
            return {"success": False, "message": f"{type(e).__name__}: {e}"}


RESTART_SELECTION_ENV = "OPENPI_SERVE_SELECTION"  # (a restart's model, over the registry's "default")
RESTART_COMPILE_ENV = "OPENPI_SERVE_COMPILE"


def _restart_with(selection: str, compile_: bool) -> None:
    """--model-registry switches: this process replaced by a fresh one (same pid, arguments and port) serving
    ``selection``. In place, the old model's memory wasn't all given back: JAX keeps its freed GPU pool for the process's
    life, so a PyTorch model loaded after a JAX one ran beside ~10 GB of it (21 GB vs 11 GB started on its own) and its
    batched candidates ran out of memory. Clients reconnect (WebsocketClientPolicy: infer / send_control retry once the
    server is back, i.e. once the model is loaded)."""
    os.environ[RESTART_SELECTION_ENV] = selection
    os.environ[RESTART_COMPILE_ENV] = "1" if compile_ else "0"
    logging.info("Restarting: %s %s", sys.executable, " ".join(sys.argv))
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _release_gpu_memory() -> None:
    """The old policy's GPU memory given back before the next one loads: only one model on the GPU at a time. A PyTorch
    model is held by torch.compile's caches and, once freed, kept in torch's caching allocator -- both cleared here (a
    JAX model's arrays go with gc; JAX keeps its freed pool for its own reuse, so a PyTorch model loaded after a JAX one
    shares the GPU with that pool)."""
    import gc
    import sys

    gc.collect()
    if "torch" in sys.modules:
        import torch

        torch._dynamo.reset()  # (the compiled sample_actions, which holds the model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            logging.info("GPU after releasing the old policy: %.1f of %.1f GiB free", free / 2**30, total / 2**30)
    import jax

    jax.clear_caches()


def main(args: Args) -> None:
    reload_handler: _ReloadHandler | None = None

    if args.model_registry is not None:  # (families of checkpoints, each with its own config: the GUI picks)
        import json

        registry = json.loads(pathlib.Path(args.model_registry).read_text())
        reload_handler = _ReloadHandler(None, None, pathlib.Path("."), args.default_prompt, registry=registry)
        selections = reload_handler._registry_selections()
        if not selections:
            raise ValueError(f"--model-registry {args.model_registry}: no checkpoints found under its roots")
        initial_selection = registry.get("default") if registry.get("default") in selections else list(selections)[-1]
        restart = os.environ.get(RESTART_SELECTION_ENV)  # (a GUI switch: _restart_with)
        if restart in selections:
            initial_selection = restart
        compile_ = os.environ.get(RESTART_COMPILE_ENV) == "1"
        logging.info("Model registry %s: %d checkpoints; serving %s%s", args.model_registry, len(selections), initial_selection,
                     " (switched to)" if restart in selections else "")
        policy = reload_handler._build_policy(initial_selection, compile=compile_)
        reload_handler._current = initial_selection
    elif args.checkpoint_root is not None:
        if not isinstance(args.policy, Checkpoint):
            raise ValueError("--checkpoint-root requires --policy.config/--policy.dir (a Checkpoint), not Default.")
        # Build the reload handler FIRST and load the initial policy through
        # its _build_policy (not the plain create_policy() path below) so the
        # very first served policy gets the same fixed-norm_stats override
        # every later hot-reload does -- otherwise, defaulting --policy.dir
        # straight to BASE_CHECKPOINT_DIR (to serve base at startup) would
        # silently use pi0.5's own generic pretraining norm stats (bundled in
        # its assets/ dir) for the FIRST policy served, while switching to
        # "base" via the GUI afterward would correctly use this task's own
        # norm stats -- same weights, two different normalizations depending
        # on how you got there. Fine-tuned checkpoints don't have this
        # problem (train.py's own save_state bundles this task's norm stats
        # into every checkpoint's assets/ dir, see checkpoints.py), but this
        # keeps initial-load and reload behavior identical regardless.
        reload_handler_train_config = _config.get_config(args.policy.config)
        initial_selection = (
            BASE_CHECKPOINT_SELECTION
            if str(args.policy.dir).rstrip("/") == BASE_CHECKPOINT_DIR
            else pathlib.Path(args.policy.dir).name
        )
        # reload_handler needs a `server` reference to call set_policy() from
        # its own control-command handling, but the server isn't constructed
        # until below -- passed in afterward via set_server, then used here
        # only for _build_policy (doesn't touch self._server yet).
        reload_handler = _ReloadHandler(
            None,
            reload_handler_train_config,
            pathlib.Path(args.checkpoint_root),
            args.default_prompt,
        )
        policy = reload_handler._build_policy(initial_selection)
        reload_handler._current = initial_selection
    else:
        policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    # (the server holds the policy from here: this frame's reference kept would pin the startup model on the GPU for
    # the server's whole life -- under every model hot-reloaded after it)
    del policy

    rl_control = None
    if args.rl_dir is not None:
        from openpi.policies import lora_rl as _lora_rl

        rl_control = _lora_rl.RLControl(lambda: server._policy, args.rl_dir)  # noqa: SLF001
        logging.info("LoRA RL control commands on (rl_*): saving under %s", args.rl_dir)

    def on_control(command: dict) -> dict:
        if rl_control is not None and str(command.get("cmd", "")).startswith("rl_"):
            return rl_control(command)
        if reload_handler is not None:
            return reload_handler(command)
        return {"success": False, "message": f"no handler for {command.get('cmd')!r}"}

    if reload_handler is not None or rl_control is not None:
        server.set_on_control(on_control)
    if reload_handler is not None:
        reload_handler._server = server
        logging.info(
            "Hot-reload enabled (%s, initial=%s)",
            f"model_registry={args.model_registry}" if args.model_registry else f"checkpoint_root={args.checkpoint_root}",
            reload_handler._current,
        )

    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
