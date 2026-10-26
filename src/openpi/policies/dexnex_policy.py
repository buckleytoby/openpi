import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_dexnex_example() -> dict:
    """Creates a random input example for the dexnex policy."""
    return {
        "observation/state": np.random.rand(10),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/env_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/overhead_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/env_image_valid": True,
        "prompt": "pick up the pellet from the tray and place it in the erlenmeyer flask",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# The palm roll and yaw: state / action indices (xyz, roll, pitch, yaw, gripper, ...)
ROLL_YAW = [3, 5]


def _rotate_roll_yaw(x, offset: float) -> np.ndarray:
    """``x`` (..., >= 6: palm xyz + rpy first) with ``offset`` added to its roll and yaw, wrapped to [-pi, pi)."""
    x = np.array(x, dtype=np.float32 if np.asarray(x).dtype != np.float64 else np.float64)
    x[..., ROLL_YAW] = (x[..., ROLL_YAW] + offset + np.pi) % (2 * np.pi) - np.pi
    return x


@dataclasses.dataclass(frozen=True)
class DexnexInputs(transforms.DataTransformFn):
    """
    Converts inputs to the model to the expected format. Used for both training and inference.

    Camera mapping (pi0/pi0.5 support one base view + two wrist views):
      - env_image      -> base_0_rgb        (third-person view; not recorded for all episodes,
                                              see env_image_valid below)
      - wrist_image    -> left_wrist_0_rgb
      - overhead_image -> right_wrist_0_rgb

    `state` is 10-dim: 6D palm pose (xyz + rpy) + 1D gripper value + 3D biotac tactile
    (thumb, index, middle). Only its first `state_dim` values reach the model: 7 drops the
    tactile (sim/real co-training: the sim has no BioTac -- its dataset rows hold zeros there);
    the inference node can keep sending all 10.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    # 10: palm pose + gripper + biotac; 7: palm pose + gripper (no tactile)
    state_dim: int = 10
    # Added to the palm roll and yaw (state and actions, wrapped to [-pi, pi); DexnexOutputs takes it back off): moves
    # their +-pi seam out of the data. -pi/2 for the dexnex data, whose palm yaw spans 60 -> 180 -> -135 deg (none in
    # -120..30) and roll ~110 deg: both continuous after it. 0: as recorded.
    angle_offset: float = 0.0

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/env_image"])
        left_wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_image = _parse_image(data["observation/overhead_image"])

        # env_image was only added partway through data collection -- episodes recorded
        # before that have an all-zero env_image, which we mask out here so the model
        # doesn't treat the zeros as real visual input. At inference time (no
        # "observation/env_image_valid" key), env_image is always real, so default to True.
        env_image_valid = bool(data.get("observation/env_image_valid", True))

        state = np.asarray(data["observation/state"])
        if self.angle_offset:
            state = _rotate_roll_yaw(state, self.angle_offset)

        inputs = {
            "state": state[..., : self.state_dim],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_ if env_image_valid else np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = _rotate_roll_yaw(data["actions"], self.angle_offset) if self.angle_offset else data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DexnexOutputs(transforms.DataTransformFn):
    """Converts outputs from the model back to the dataset-specific format. Used for inference only."""

    # DexnexInputs.angle_offset: taken back off the predicted roll and yaw (the robot's convention again)
    angle_offset: float = 0.0

    def __call__(self, data: dict) -> dict:
        # Actions were padded to the model action dimension; only the first 7 (palm pose +
        # gripper) are real -- the rest is padding.
        actions = np.asarray(data["actions"][..., :7])
        if self.angle_offset:
            actions = _rotate_roll_yaw(actions, -self.angle_offset)
        return {"actions": actions}
