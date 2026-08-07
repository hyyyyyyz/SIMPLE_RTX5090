"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

DreamZero G1 Whole-Body baseline adapter — Decoupled WBC variant.

This is the decoupled-WBC counterpart of `simple.baselines.dreamzero`. It
talks to the same DreamZero policy server (same /act, /config, /flush
protocol; same Psi0-flat 36-D action layout) but emits `vla_cmd`
ActionCmds and converts them through the Sonic decoupled-WBC stack
(see `simple.agents.sonic_decoupled_wbc_agent.SonicDecoupledWbcAgent`),
mirroring `Psi0DecoupledWbcAgent`.

Use this agent for tasks that run on the G1Sonic robot (e.g.
`G1WholebodyXMovePickTeleop-v0` evaluated under the decoupled-WBC eval
CLI), and use `DreamzeroAgent` for the eval_move_actuators-driven
G1Wholebody envs.

The 36-D action layout and 32-D state packing are identical to
`dreamzero.py`; see that file's docstring for the full breakdown.

Copyright (c) 2025 USC PSI Lab and Contributors.
"""

import os
import time

import numpy as np
import requests

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd
from simple.baselines.client import HttpActionClient
from simple.baselines.dreamzero import STATE_SLICES, from_dreamzero_upper_joints


class DreamzeroDecoupledWbcAgent(SonicDecoupledWbcAgent):
    """Client for a remote DreamZero G1 policy server, decoupled-WBC variant.

    Differences vs. `DreamzeroAgent`:
      - Inherits from `SonicDecoupledWbcAgent` so each emitted action chunk
        is converted through the WBC policy into a `decoupled_wbc` ActionCmd
        consumable by the G1Sonic robot.
      - No "stand warm-up" loop — the WBC stack handles balance on its own.
      - Tracks the joint name order via `_dwbc_robot_model.get_joint_group_indices`
        so `target_upper_body_pose` is populated in the right key order.

    Identical to `DreamzeroAgent`:
      - Server /config query for action_horizon / video stride.
      - Frame buffering across chunks for the world-model conditioning.
      - DREAMZERO_ACTION_HORIZON env override.
      - dataset="dreamzero_g1_simple" routing on the server side.
      - Receding-horizon execution of the first action_horizon actions only.
      - /flush on reset to commit predicted-video latents.
    """

    def __init__(
        self,
        robot,
        host: str,
        port: int,
        upsample_factor: int = 1,
        action_horizon: int = 24,
        client=None,
        **kwargs,
    ):
        super().__init__(robot, **kwargs)

        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = client or HttpActionClient(self.server_ip, self.server_port)

        # Wait for server, then query model config so frame collection
        # matches training. Both containers are often launched in parallel —
        # don't give up on /config just because the server is still warming.
        self._wait_for_server()
        self._server_config = self._query_server_config()
        self.action_horizon = int( # prioitize "action_horizon" in the server config
            str(self._server_config.get("action_horizon", action_horizon)),
        )

        self._video_frames_per_chunk = self._server_config.get("video_frames_per_chunk", 8)
        self._video_stride = self._server_config.get("video_stride", self.action_horizon // 8)

        self._global_step_idx = 0
        self._session_idx = 0
        self._session_id = self._make_session_id()
        self._obs_frame_buffer: list[np.ndarray] = []

        # last command (high level input to lower policy)
        self._last_base_height_cmd = 0.74  # FIXME hardcoded for g1 wholebody, need to be more general in the future
        self._reset_history = True

        # Resolve the joint name order the WBC policy expects for the upper body.
        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [
            name for name, idx in self._dwbc_robot_model.joint_to_dof_index.items()
            if idx in indices
        ]

    def _wait_for_server(self, timeout_s: float = 900.0, poll_interval_s: float = 5.0) -> None:
        """Block until the server /health returns 200 or timeout elapses.

        Server startup (shard load + DeepSpeed init + compile warmup) can take
        3-5 min for the 14B full-ft model. Simulator init on the client side
        is comparable, so launching both in parallel is the usual workflow —
        wait here rather than giving up on the first failed /config.
        """
        url = f"http://{self.server_ip}:{self.server_port}/health"
        t0 = time.time()
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    elapsed = time.time() - t0
                    print(f"[DreamzeroDecoupledWbcAgent] /health ok after {elapsed:.1f}s ({attempt} attempts)")
                    return
            except Exception:
                pass
            elapsed = time.time() - t0
            if elapsed > timeout_s:
                print(f"[DreamzeroDecoupledWbcAgent] /health still not responding after {elapsed:.0f}s — giving up, proceeding with defaults")
                return
            if attempt == 1 or attempt % 6 == 0:
                print(f"[DreamzeroDecoupledWbcAgent] waiting for server at {url} ... ({elapsed:.0f}s elapsed)")
            time.sleep(poll_interval_s)

    def _query_server_config(self) -> dict:
        try:
            resp = requests.get(f"http://{self.server_ip}:{self.server_port}/config", timeout=5)
            if resp.status_code == 200:
                cfg = resp.json()
                print(f"[DreamzeroDecoupledWbcAgent] server config: {cfg}")
                return cfg
        except Exception as e:
            print(f"[DreamzeroDecoupledWbcAgent] could not query /config: {e}, using defaults")
        return {}

    def _make_session_id(self) -> str:
        return f"dreamzero-dwbc-{os.getpid()}-{self._session_idx}"

    def get_action(
        self,
        observation,
        instruction=None,
        info=None,
        conditions=None,
        **kwargs,
    ):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        # Always buffer observation frames for the next server query.
        self._obs_frame_buffer.append(observation["head_stereo_left"])

        if len(self._action_queue) == 0:
            # Subsample frames at video_stride aligned to the END so the most
            # recent observation is always included (server takes the last
            # num_frame_per_block latents from VAE output).
            buf = self._obs_frame_buffer
            if len(buf) <= 1:
                frames = list(buf)
            else:
                n_want = self._video_frames_per_chunk
                stride = self._video_stride
                indices = [len(buf) - 1 - i * stride for i in range(n_want)]
                indices = [max(0, idx) for idx in reversed(indices)]
                frames = [buf[i] for i in indices]
            video = np.stack(frames, axis=0)  # (T, H, W, 3)
            self._obs_frame_buffer = self._obs_frame_buffer[-1:]

            observations = {
                "rgb_head_stereo_left": video,
            }

            # Reconstruct 32D Psi0-flat state from 43D env qpos + last torso cmd.
            proprio = observation["joint_qpos"][None]
            waist_rpy = proprio[:, [13, 14, 12]]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES] + [
                    waist_rpy,
                    np.array([[self._last_base_height_cmd]], dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32)  # shape (1, 32)
            state_dict = {"states": states}

            # Tell server to reset its KV cache at episode boundary.
            history = {
                "session_id": self._session_id,
                "episode_index": int(info.get("episode_index", -1)) if info is not None else -1,
                "step_index": int(self._global_step_idx),
            }
            if self._reset_history:
                history["reset"] = True
                self._reset_history = False

            pred_action, *_ = self.client.query_action(
                observations,
                instruction or "perform the task",
                state_dict,
                {},
                history=history,
                dataset="dreamzero_g1_simple",
            )
            n_execute = min(self.action_horizon, pred_action.shape[0])
            print(f"[DreamzeroDecoupledWbcAgent] received chunk of {pred_action.shape[0]} actions, executing {n_execute}")

            for i in range(n_execute):
                for _ in range(self.upsample_factor):
                    target_qpos = dict(zip(
                        self.robot.joint_names[15:],
                        from_dreamzero_upper_joints(pred_action[i][:28]),
                    ))
                    target_waist_qpos = {
                        "waist_roll_joint":  pred_action[i][28],
                        "waist_pitch_joint": pred_action[i][29],
                        "waist_yaw_joint":   pred_action[i][30],
                    }
                    self.queue_action(ActionCmd(
                        "vla_cmd",
                        target_upper_body_pose={**target_qpos, **target_waist_qpos},
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
        # Flush any unsaved predicted-video latents on the server so the last
        # episode's video is written to disk before we start a new session.
        try:
            requests.post(f"http://{self.server_ip}:{self.server_port}/flush", timeout=30)
        except Exception:
            pass
        self._global_step_idx = 0
        self._session_idx += 1
        self._session_id = self._make_session_id()
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._obs_frame_buffer = []
        self._last_base_height_cmd = 0.74
