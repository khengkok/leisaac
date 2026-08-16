# /// script
# requires-python = ">=3.10"
# dependencies = ["pyzmq", "openvr"]
# ///
"""Publish a Pico 4 Ultra controller's pose + trigger/grip state over ZMQ PUB.

Run this on the PC connected to the Pico 4 Ultra via PICO Connect / Streaming
Assistant, which exposes the headset and controllers as a SteamVR device.
This script reads one controller's pose via OpenVR and publishes it; a remote
LeIsaac instance (teleop_device=pico) subscribes to receive it for
teleoperation, exactly like so101_joint_state_server.py does for the SO101
leader arm.

Prerequisites:
    - SteamVR running, with the Pico headset connected and tracked via
      PICO Connect / Streaming Assistant.
    - pip install pyzmq openvr

Usage:
    python pico_pose_server.py --hand right --rate 90
"""

import argparse
import struct
import time

import openvr
import zmq


def find_controller(vr_system, hand: str):
    role = openvr.TrackedControllerRole_RightHand if hand == "right" else openvr.TrackedControllerRole_LeftHand
    for device_index in range(openvr.k_unMaxTrackedDeviceCount):
        if vr_system.getTrackedDeviceClass(device_index) != openvr.TrackedDeviceClass_Controller:
            continue
        if vr_system.getControllerRoleForTrackedDeviceIndex(device_index) == role:
            return device_index
    return None


def pose_to_pos_quat(pose_matrix):
    """Convert an OpenVR 3x4 pose matrix to (position, quaternion[xyzw])."""
    m = pose_matrix
    px, py, pz = m[0][3], m[1][3], m[2][3]

    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]

    trace = r00 + r11 + r22
    if trace > 0:
        s = 0.5 / (trace + 1.0) ** 0.5
        qw = 0.25 / s
        qx = (r21 - r12) * s
        qy = (r02 - r20) * s
        qz = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * (1.0 + r00 - r11 - r22) ** 0.5
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * (1.0 + r11 - r00 - r22) ** 0.5
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = 2.0 * (1.0 + r22 - r00 - r11) ** 0.5
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    return (px, py, pz), (qx, qy, qz, qw)


def main():
    parser = argparse.ArgumentParser(description="Pico 4 Ultra controller pose publisher (via SteamVR)")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="Which controller to publish")
    parser.add_argument("--bind", default="tcp://0.0.0.0:5557")
    parser.add_argument("--rate", type=int, default=90, help="Publish rate in Hz")
    args = parser.parse_args()

    print("Connecting to SteamVR (make sure PICO Connect / Streaming Assistant is running)...")
    vr_system = openvr.init(openvr.VRApplication_Other)

    print(f"Waiting for the {args.hand} Pico controller to be tracked...")
    device_index = None
    while device_index is None:
        device_index = find_controller(vr_system, args.hand)
        if device_index is None:
            time.sleep(0.5)
    print(f"Found {args.hand} controller at device index {device_index}")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.CONFLATE, 1)
    pub.bind(args.bind)
    print(f"Publishing on {args.bind} at {args.rate} Hz")
    time.sleep(0.5)

    interval = 1.0 / args.rate
    count = 0
    next_t = time.monotonic()

    try:
        while True:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            pose = poses[device_index]
            _, state = vr_system.getControllerState(device_index)
            trigger = state.rAxis[1].x  # analog trigger, 0.0-1.0
            grip = 1.0 if bool(state.ulButtonPressed & (1 << openvr.k_EButton_Grip)) else 0.0

            if pose.bPoseIsValid:
                (px, py, pz), (qx, qy, qz, qw) = pose_to_pos_quat(pose.mDeviceToAbsoluteTracking)
                valid = 1.0
            else:
                px = py = pz = qx = qy = qz = 0.0
                qw = 1.0
                valid = 0.0

            pub.send(struct.pack("<10f", px, py, pz, qx, qy, qz, qw, trigger, grip, valid), zmq.NOBLOCK)

            count += 1
            if count % (args.rate * 10) == 0:
                print(f"{count} msgs sent")

            next_t += interval
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        print(f"\nDone: {count} msgs")
    finally:
        openvr.shutdown()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
