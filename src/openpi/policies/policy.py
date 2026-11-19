from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


# (dexnex) torch.compile's mode when serving a PyTorch policy: "default" -- inductor without max-autotune's kernel
# benchmarking (that one took minutes to compile on the first call); training keeps its config's mode
SERVE_COMPILE_MODE = "default"


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        # (dexnex: the actions' normalization, for "__step_offset" -- an offset given in the action units)
        unnorm = [t for t in output_transforms if isinstance(t, _transforms.Unnormalize)]
        self._action_norm = ((unnorm[0].norm_stats or {}).get("actions"), unnorm[0].use_quantiles) if unnorm else (None, False)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        # (dexnex) LoRA RL (lora_rl.LoRATrainer.record): every batched call's model-space observation and chunks
        # recorded; its id returned as "__rl_id"
        self.rl_recorder = None
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            # (dexnex: compiled or eager per call -- an inference's "__compile"; without one, as loaded. torch.compile
            # only wraps: the compile itself happens on the compiled function's first call)
            self._sample_eager = getattr(model, "sample_actions_eager", model.sample_actions)
            self._sample_compiled = model.sample_actions if getattr(model, "compile_mode", None) else None
            self._compile_default = self._sample_compiled is not None
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_samples", "offset_steps"))
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        # (dexnex) "__num_samples": N > 1 samples N action chunks for this one observation in one call -- the model
        # encodes the observation's prefix once and denoises N independent noise samples as a batch (Pi0.sample_actions'
        # num_samples); "actions" is then [N, H, D].
        num_samples = int(inputs.pop("__num_samples", 1))
        # (dexnex) "__noise": the denoising's starting noise [N, H, D_model] (e.g. a returned "__noise", to re-denoise the
        # same sample); "__step_offset" [H, d] in the action units (d <= D_model: the first d dims) with "__offset_steps"
        # k: added in shares over the last k denoising steps (Pi0.sample_actions' step_offset). The noise used is
        # returned as "__noise".
        given_noise = inputs.pop("__noise", None)
        step_offset = inputs.pop("__step_offset", None)
        offset_steps = int(inputs.pop("__offset_steps", 0))
        # (dexnex: the drawn starting noise x this -- wider candidates; one value, or one per sample [num_samples])
        noise_temperature = np.asarray(inputs.pop("__noise_temperature", 1.0), dtype=np.float32).reshape(-1)
        compile_request = inputs.pop("__compile", None)  # (dexnex: PyTorch -- torch.compile on / off for this call)
        if self._is_pytorch_model:
            self._sample_actions = self._pick_sample_fn(compile_request)
        if given_noise is not None and noise is None:
            noise = np.asarray(given_noise, dtype=np.float32)
        inputs = self._input_transform(inputs)
        model_inputs = inputs  # (numpy, model space: the LoRA RL record's observation)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            # (one observation: num_samples shares its encoded prefix -- Pi0Pytorch.sample_actions' num_samples)
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if num_samples > 1:
            sample_kwargs["num_samples"] = num_samples
        if noise is None:  # (drawn here, so it can be returned -- and the temperatures applied)
            horizon, dim = self._action_shape()
            if self._is_pytorch_model:  # (numpy: a JAX draw here would claim GPU memory beside the PyTorch model)
                noise = np.random.default_rng().standard_normal((num_samples, horizon, dim)).astype(np.float32)
            else:
                self._rng, noise_rng = jax.random.split(self._rng)
                noise = np.asarray(jax.random.normal(noise_rng, (num_samples, horizon, dim)))
            if noise_temperature.size not in (1, num_samples):
                raise ValueError(f"__noise_temperature: {noise_temperature.size} values for {num_samples} samples")
            noise = noise * (noise_temperature[:, None, None] if noise_temperature.size > 1 else noise_temperature[0])
        used_noise = None if noise is None else np.asarray(noise, dtype=np.float32).reshape(-1, *np.shape(noise)[-2:])
        if step_offset is not None and offset_steps > 0 and not self._is_pytorch_model:
            sample_kwargs["step_offset"] = jnp.asarray(self._offset_to_model(np.asarray(step_offset, np.float32), num_samples))
            sample_kwargs["offset_steps"] = offset_steps
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        start_time = time.monotonic()
        actions, chunk = self._sample_in_chunks(sample_rng_or_pytorch_device, inputs, sample_kwargs, num_samples)
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        to_np = (lambda x: np.asarray(x.detach().cpu())) if self._is_pytorch_model else np.asarray
        batched = jax.tree.map(to_np, outputs)
        if num_samples == 1:
            outputs = self._output_transform(jax.tree.map(lambda x: x[0, ...], batched))
        else:  # (the output transform is per sample: unnormalize etc.; "state" stays the one observation's)
            state = batched["state"]
            per = [self._output_transform({"state": state[min(i, len(state) - 1)], "actions": batched["actions"][i]})
                   for i in range(num_samples)]
            outputs = {"actions": np.stack([p["actions"] for p in per])}
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if chunk < num_samples:  # (dexnex: the samples didn't fit on the GPU at once -- run in chunks of this many)
            outputs["__chunk_size"] = chunk
        if used_noise is not None:
            outputs["__noise"] = used_noise
        if self.rl_recorder is not None and num_samples > 1:
            rl_id = self.rl_recorder(model_inputs, batched["actions"], num_samples)
            if rl_id is not None:
                outputs["__rl_id"] = rl_id
        return outputs

    def _action_shape(self) -> tuple[int, int]:
        """(action_horizon, action_dim) of the model (the PyTorch one keeps them in its config)."""
        m = self._model
        if hasattr(m, "action_horizon"):
            return m.action_horizon, m.action_dim
        return m.config.action_horizon, m.config.action_dim

    def _pick_sample_fn(self, compile_request):
        """(dexnex, PyTorch) The compiled sample_actions if this call asks for it (default: as the model was loaded), else
        the eager one. The compiled one is made on first use (its first call compiles: slow; later calls reuse it)."""
        want = self._compile_default if compile_request is None else bool(np.asarray(compile_request))
        if not want:
            return self._sample_eager
        if self._sample_compiled is None:
            mode = SERVE_COMPILE_MODE
            logging.info("torch.compile on (mode %s): the next call compiles -- slow once", mode)
            self._sample_compiled = torch.compile(self._sample_eager, mode=mode)
        return self._sample_compiled

    def _sample_in_chunks(self, rng, inputs, sample_kwargs: dict, num_samples: int):
        """(dexnex) The model's sample_actions for num_samples samples; out of GPU memory, in chunks run one after another
        (halved until they fit; the size that fit is kept for later calls). Returns (actions [N, H, D] -- [1, H, D] for one
        sample -- and the chunk size used)."""
        chunk = min(num_samples, getattr(self, "_max_chunk", num_samples))
        while True:
            try:
                parts = []
                for start in range(0, num_samples, chunk):
                    n = min(chunk, num_samples - start)
                    kwargs = dict(sample_kwargs)
                    for key in ("noise", "step_offset"):  # (per sample: this chunk's)
                        if key in kwargs and kwargs[key].shape[0] == num_samples:
                            kwargs[key] = kwargs[key][start:start + n]
                    if "num_samples" in kwargs:  # (one observation, its prefix shared by the chunk's samples)
                        kwargs["num_samples"] = n
                    parts.append(self._sample_actions(rng, _model.Observation.from_dict(inputs), **kwargs))
                break
            except Exception as e:  # noqa: BLE001
                text = f"{type(e).__name__}: {e}"
                if chunk == 1 or not ("OutOfMemory" in text or "out of memory" in text or "RESOURCE_EXHAUSTED" in text):
                    raise
                parts = None
                if self._is_pytorch_model:
                    torch.cuda.empty_cache()
                chunk = (chunk + 1) // 2
                logging.warning("%d samples: out of GPU memory -- retrying in chunks of %d", num_samples, chunk)
        if chunk < num_samples:
            self._max_chunk = chunk
        if len(parts) == 1:
            return parts[0], chunk
        return (torch.cat(parts) if self._is_pytorch_model else jnp.concatenate(parts)), chunk

    def _offset_to_model(self, offset: np.ndarray, num_samples: int) -> np.ndarray:
        """An offset [H, d] in the action units -> the model's normalized action space [num_samples, H, D_model] (a
        difference: only the normalization's scale applies -- 2 / (q99 - q01) or 1 / std)."""
        stats, quantiles = self._action_norm
        if stats is None:
            raise ValueError("__step_offset: no action normalization stats in the policy")
        d = offset.shape[-1]
        scale = 2.0 / (np.asarray(stats.q99)[:d] - np.asarray(stats.q01)[:d] + 1e-6) if quantiles else 1.0 / (np.asarray(stats.std)[:d] + 1e-6)
        full = np.zeros((num_samples, *self._action_shape()), np.float32)
        full[:, : offset.shape[0], :d] = offset * scale
        return full

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
