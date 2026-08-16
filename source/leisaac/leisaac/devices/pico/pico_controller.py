"""Pico 4 Ultra VR controller teleop device.

Reads a controller's 6-DoF pose + trigger/grip state streamed over ZMQ from
``pico_pose_server.py`` (run on the PC connected to the headset via PICO
Connect / Streaming Assistant, which exposes it as a SteamVR device). Produces
the same 8-dim delta-pose action used by :class:`SO101Keyboard` /
:class:`SO101Gamepad`, so it is a drop-in ``keyboard``/``gamepad``-style
device for single-arm tasks -- no USB/cable to the EC2 machine needed.
"""

import struct
import threading

import numpy as np

from ..device_base import Device

# Best-guess mapping from the Pico controller's own local axes (OpenVR convention:
# +X = right, +Y = up, -Z = forward/pointing direction) to this repo's gripper
# target-frame convention (see SO101Keyboard: +X = up, -Z = forward). Z is already
# consistent between the two conventions (both use -Z as "forward"), so only X/Y
# are swapped here. This has NOT been empirically verified against the running
# sim -- if the gripper moves in an unexpected direction relative to your hand,
# adjust the signs below while watching the viewport.
_CONTROLLER_TO_GRIPPER_AXES = np.array(
    [
        [0.0, 1.0, 0.0],  # gripper +X (up)    <- controller +Y (up)
        [-1.0, 0.0, 0.0],  # gripper +Y         <- controller -X (left)
        [0.0, 0.0, 1.0],  # gripper +Z (back)  <- controller +Z (back)
    ]
)

_MAX_POS_DELTA = 0.05  # meters/tick safety clamp against tracking glitches
_MAX_ROT_DELTA = 0.3  # radians/tick safety clamp against tracking glitches
_GRIP_THRESHOLD = 0.5


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v (3,) by quaternion q (xyzw)."""
    qv = np.array([v[0], v[1], v[2], 0.0])
    return _quat_multiply(_quat_multiply(q, qv), _quat_conjugate(q))[:3]


def _quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to intrinsic XYZ Euler angles (roll, pitch, yaw)."""
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


class SO101PicoController(Device):
    """A Pico 4 Ultra controller for sending SE(3) commands as delta poses for so101 single arm.

    Controls:
        Hold GRIP  - engage tracking (clutch). While held, moving/rotating the
                     controller moves/rotates the gripper by the same delta,
                     scaled by ``sensitivity``. Release GRIP to reposition your
                     hand without moving the robot.
        TRIGGER    - gripper close (squeezed) / open (released), proportional
                     to how far the trigger is moved each tick.
        B / R / N  - same as every other device (start / reset-fail / reset-success).
    """

    _MSG_FORMAT = "<10f"  # px, py, pz, qx, qy, qz, qw, trigger, grip, valid

    def __init__(self, env, endpoint: str = "tcp://localhost:5557", sensitivity: float = 1.0):
        self._endpoint = endpoint
        self._connected = False
        self._ctx = None
        self._sub = None
        self._lock = threading.Lock()
        self._cached = {
            "pos": np.zeros(3),
            "quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "trigger": 0.0,
            "grip": 0.0,
            "valid": 0.0,
        }
        self._recv_thread = None

        super().__init__(env, "pico")

        # store inputs (Pico gives real controller-space meters/radians, so unlike
        # the keyboard/gamepad fixed per-tick step, this scale is close to 1:1)
        self.pos_sensitivity = 1.0 * sensitivity
        self.rot_sensitivity = 1.0 * sensitivity
        self.gripper_sensitivity = 1.0 * sensitivity

        # command buffer (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)

        # clutch/reference state
        self._have_ref = False
        self._ref_pos = np.zeros(3)
        self._ref_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._have_prev_trigger = False
        self._prev_trigger = 0.0
        self._tick_count = 0

        # initialize the target frame
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "gripper"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

        self.connect()

    def _recv_loop(self):
        count = 0
        while self._connected:
            try:
                msg = self._sub.recv(flags=0)  # blocks here, no lock held
                px, py, pz, qx, qy, qz, qw, trigger, grip, valid = struct.unpack(self._MSG_FORMAT, msg)
                with self._lock:
                    self._cached = {
                        "pos": np.array([px, py, pz]),
                        "quat": np.array([qx, qy, qz, qw]),
                        "trigger": trigger,
                        "grip": grip,
                        "valid": valid,
                    }
                count += 1
                if count == 1:
                    print(f"[pico] first message received: pos=({px:.3f},{py:.3f},{pz:.3f}) valid={valid:.0f}")
                elif count % 60 == 0:
                    print(
                        f"[pico] #{count} pos=({px:.3f},{py:.3f},{pz:.3f}) trigger={trigger:.2f} "
                        f"grip={grip:.2f} valid={valid:.0f}"
                    )
            except Exception as e:
                print(f"[pico] recv loop stopped: {e}")
                break

    def connect(self):
        import zmq

        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect(self._endpoint)
        self._connected = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        print(f"SO101-Pico-Controller connected to {self._endpoint}")

    def disconnect(self):
        self._connected = False
        if self._recv_thread:
            self._recv_thread.join(timeout=2)
        if self._sub:
            self._sub.close()
        if self._ctx:
            self._ctx.term()
        print("SO101-Pico-Controller disconnected.")

    def _add_device_control_description(self):
        self._display_controls_table.add_row(["GRIP (hold)", f"engage tracking, endpoint: {self._endpoint}"])
        self._display_controls_table.add_row(["TRIGGER", "gripper close/open (proportional)"])

    def get_device_state(self):
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        self._delta_action[:] = 0.0
        self._have_ref = False
        self._have_prev_trigger = False

    def advance(self):
        self._update_action()
        self._tick_count += 1
        if self._tick_count % 60 == 0:
            print(f"[pico] started(B pressed)={self.started} delta={np.round(self._delta_action, 4).tolist()}")
        return super().advance()

    def _update_action(self):
        with self._lock:
            state = dict(self._cached)

        self._delta_action[:] = 0.0
        self._update_gripper_action(state)
        self._update_arm_action(state)

    def _update_gripper_action(self, state):
        valid = state["valid"] > 0.5
        trigger = float(state["trigger"])
        if not valid:
            self._have_prev_trigger = False
            return
        if not self._have_prev_trigger:
            self._prev_trigger = trigger
            self._have_prev_trigger = True
            return
        trigger_delta = trigger - self._prev_trigger
        self._prev_trigger = trigger
        # squeezing the trigger further (trigger increasing) closes the gripper
        self._delta_action[7] = -trigger_delta * self.gripper_sensitivity

    def _update_arm_action(self, state):
        valid = state["valid"] > 0.5
        grip_engaged = state["grip"] > _GRIP_THRESHOLD
        pos = state["pos"]
        quat = state["quat"]

        if not valid or not grip_engaged:
            if self._have_ref:
                print(f"[pico] clutch disengaged (valid={valid}, grip={state['grip']:.2f})")
            self._have_ref = False
            return

        if not self._have_ref:
            # fresh engage (or regained tracking): take a reference sample, no motion yet
            print(f"[pico] clutch engaged at pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            self._ref_pos = pos
            self._ref_quat = quat
            self._have_ref = True
            return

        # delta expressed in the controller's own previous-frame axes, i.e. "move
        # this far along however the controller is currently pointing/oriented"
        world_delta_pos = pos - self._ref_pos
        local_delta_pos = _quat_apply(_quat_conjugate(self._ref_quat), world_delta_pos)

        local_delta_quat = _quat_multiply(_quat_conjugate(self._ref_quat), quat)
        local_delta_rot = _quat_to_euler_xyz(local_delta_quat)

        gripper_delta_pos = _CONTROLLER_TO_GRIPPER_AXES @ local_delta_pos * self.pos_sensitivity
        gripper_delta_rot = _CONTROLLER_TO_GRIPPER_AXES @ local_delta_rot * self.rot_sensitivity

        gripper_delta_pos = np.clip(gripper_delta_pos, -_MAX_POS_DELTA, _MAX_POS_DELTA)
        gripper_delta_rot = np.clip(gripper_delta_rot, -_MAX_ROT_DELTA, _MAX_ROT_DELTA)

        self._delta_action[0:3] = gripper_delta_pos
        self._delta_action[3:6] = gripper_delta_rot

        # advance the reference so next tick measures a fresh incremental delta
        self._ref_pos = pos
        self._ref_quat = quat
