"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import io
import time
import base64
import numpy as np
import requests
from PIL import Image

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd

# Ego-frame resize target sent to the Cosmos server (matches cosmos_http_client.py).
DEFAULT_IMAGE_SIZE = 256

STATE_SLICES = [ # shoule be consistent with scripts/postprocess_psi0.py
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
]

def from_psi0_upper_joints(psi0_action):
    return np.concatenate([
        psi0_action[14:28],
        psi0_action[0:3], # left thumb
        psi0_action[5:7], # left index
        psi0_action[3:5], # left middle
        psi0_action[7:14], # right hand
    ])


def _frame_to_png_b64(frame_rgb: np.ndarray) -> str:
    """[H,W,3] uint8 RGB -> base64 PNG (matches cosmos_http_client._frame_to_png_b64)."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_rgb).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CosmosPredictClient:
    """HTTP client for the Cosmos action-policy server's ``/predict`` endpoint.

    Mirrors ``cosmos_http_client.CosmosPolicyClient``: each ``predict()`` POSTs a
    single ego frame + proprio state to ``/predict`` and returns the raw action
    chunk as ``(T, D)`` float32. The payload is plain JSON (no numpy base64
    serialization) — the image is a base64 PNG and the state is a flat float list.
    """

    def __init__(self, server_ip: str, server_port: int, domain_name: str = "g1_simple",
                 image_size: int = DEFAULT_IMAGE_SIZE, view_point: str = "ego_view",
                 timeout: int = 1200):
        self._server = f"http://{server_ip}:{server_port}"
        self._domain_name = domain_name
        self._image_size = image_size
        self._view_point = view_point
        self._timeout = timeout

    def info(self):
        r = requests.get(self._server + "/info", timeout=30)
        r.raise_for_status()
        return r.json()

    def predict(self, frame_rgb: np.ndarray, prompt: str, state, history: dict = {}) -> np.ndarray:
        """POST ``/predict`` and return the predicted action chunk ``(T, D)`` float32.

        frame_rgb: (H, W, 3) uint8 RGB ego frame.
        prompt:    instruction string.
        state:     proprio conditioning vector (flattened to a float list).
        history:   dictionary containing additional information for the server (e.g., reset flag).
        """
        payload = {
            "image": _frame_to_png_b64(frame_rgb),
            "prompt": prompt,
            "domain_name": self._domain_name,
            "image_size": self._image_size,
            "view_point": self._view_point,
            "state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
            **history
        }
        resp = requests.post(self._server + "/predict", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        return np.asarray(data["action"], dtype=np.float32)  # (T, D)


class Cosmos3DecoupledWbcAgent(SonicDecoupledWbcAgent):
    def __init__(self, robot, host: str, port: int, upsample_factor=1,
                 domain_name: str = "g1_simple", image_size: int = DEFAULT_IMAGE_SIZE, **kwargs):
        super().__init__(robot, **kwargs)

        self.server_ip = host # if access server host inside docker container
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = CosmosPredictClient(
            self.server_ip, self.server_port,
            domain_name=domain_name, image_size=image_size,
        )
        self._global_step_idx = 0

        # last command (high level input to lower policy)
        self._last_base_height_cmd = 0.74  # FIXME hardcoded for g1 wholebody, need to be more general in the future
        self._reset_history = True

        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [name for name, idx in self._dwbc_robot_model.joint_to_dof_index.items() if idx in indices]

    def get_action(
        self,
        observation,
        instruction=None,
        info=None,
        conditions=None,
        **kwargs
    ):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if len(self._action_queue) == 0:
            # Assemble the ego frame (RGB) and the proprio conditioning state, then
            # query the Cosmos server via its /predict endpoint (see cosmos_http_client.py).
            frame_rgb = np.asarray(observation["head_stereo_left"], dtype=np.uint8)  # [H, W, 3] RGB

            proprio = observation["joint_qpos"][None]
            waist_rpy = proprio[:, [13, 14, 12]]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES] + [
                    waist_rpy,
                    np.array([[self._last_base_height_cmd]], dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32) # (1, 32)

            if self._reset_history:
                history = {"reset": True}
                self._reset_history = False
            else:
                history = {}

            pred_action = self.client.predict(
                frame_rgb,
                instruction or "bend to pick up the object",
                states,
                history
            )  # (T, D)
            print(f"Received {pred_action.shape[0]} actions from server.")
            for i in range(pred_action.shape[0]):
                for _ in range(self.upsample_factor): # account for upsampling during training
                    target_qpos = dict(
                        zip(
                            self.robot.joint_names[15:],
                            from_psi0_upper_joints(pred_action[i][:28])
                        )
                    )
                    target_waist_qpos = {
                        "waist_yaw_joint": pred_action[i][30],
                        "waist_roll_joint": pred_action[i][28],
                        "waist_pitch_joint": pred_action[i][29]
                    }
                    self.queue_action(ActionCmd(
                        "vla_cmd",
                        target_upper_body_pose={**target_qpos, **target_waist_qpos},  # (31,)
                        navigate_cmd=pred_action[i][32:36],
                        base_height_command=pred_action[i][31:32],
                    ))

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type == "vla_cmd":

            proprio = self.robot.prepare_obs()
            wbc_obs = self._build_wbc_observation(proprio)
            self._wbc_policy.set_observation(wbc_obs)
            t_now = time.monotonic()

            control_freq = self._control_frequency
            target_time = t_now + 1 / control_freq

            target_upper_body_pose = np.array([
                action_cmd["target_upper_body_pose"][jName] for jName in self.sonic_upper_joint_names
            ], dtype=np.float32)
            goal = {
                "target_upper_body_pose": target_upper_body_pose,
                "navigate_cmd": action_cmd["navigate_cmd"],
                "base_height_command": action_cmd["base_height_command"],
                "target_time": target_time,
                "interpolation_garbage_collection_time": t_now - 2 / control_freq,
                "timestamp": t_now,
            }
            self._wbc_policy.set_goal(goal)
            self._last_base_height_cmd = float(action_cmd["base_height_command"][0])
            wbc_action = self._wbc_policy.get_action(time=t_now)
            self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
            self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
            self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

            # createa a new ActionCmd for the g1_sonic robot
            action_cmd = ActionCmd(
                "decoupled_wbc",
                target_q=self._cached_target_q,
                left_hand_q=self._cached_left_hand_q,
                right_hand_q=self._cached_right_hand_q,
            )
        else:
            raise ValueError(f"Unexpected action type {action_cmd.type} from queue.")

        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd

    def reset(self, **kwargs):
        super().reset(**kwargs)  # clear queue

        self._global_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_base_height_cmd = 0.74
