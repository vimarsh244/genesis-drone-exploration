import argparse
import os
import threading

import numpy as np

import genesis as gs
from genesis.utils.geom import euler_to_quat

"""
LiDAR Teleop Drone - Interactive drone control with LiDAR/Depth sensor visualization

This script demonstrates real-time drone teleoperation with LiDAR point visualization
using the Genesis physics engine. The drone can be controlled via keyboard inputs to
move around obstacles while visualizing sensor data.

Features:
  - Real-time keyboard-based drone teleop control (RPM-based)
  - LiDAR point cloud visualization or depth camera visualization
  - Multiple sensor patterns: spherical, grid, or depth
  - Obstacle-rich environment with cylinders and boxes
  - Multi-environment simulation support
  - GPU or CPU backend selection

Usage:
  Basic usage with spherical LiDAR pattern:
    python lidar_teleop_drone.py

  With depth camera visualization:
    python lidar_teleop_drone.py --pattern depth

  With grid pattern LiDAR:
    python lidar_teleop_drone.py --pattern grid

  With multiple environments (B=4 parallel environments):
    python lidar_teleop_drone.py -B 4

  CPU backend:
    python lidar_teleop_drone.py --cpu

Keyboard Controls:
  [↑/↓/←/→]: Move forward/backward/left/right
  [space]: Increase thrust (all rotors spin faster)
  [shift]: Decrease thrust (all rotors spin slower)
  [\\]: Reset to hover
  [esc]: Quit

Notes:
  - Requires pynput library for keyboard input: pip install pynput
  - Drone model: Crazyflie 2.X quadcopter (cf2x.urdf)
  - Hover RPM: ~14468.43
"""

IS_PYNPUT_AVAILABLE = False
try:
    from pynput import keyboard

    IS_PYNPUT_AVAILABLE = True
except ImportError:
    pass

# drone control parameters
DRONE_HOVER_RPM = 14468.429183500699
DRONE_ROTATION_DELTA = 500.0  # rpm change for rotation control
DRONE_THRUST_DELTA = 50.0  # rpm change for thrust control

# box/go2 keyboard teleop parameters
KEY_DPOS = 0.1
KEY_DANGLE = 0.1

# Movement when no keyboard control is available
MOVE_RADIUS = 1.0
MOVE_RATE = 1.0 / 100.0

# Number of obstacles to create in a ring around the robot
NUM_CYLINDERS = 8
NUM_BOXES = 6
CYLINDER_RING_RADIUS = 3.0
BOX_RING_RADIUS = 5.0


class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        try:
            self.listener.stop()
        except NotImplementedError:
            # Dummy backend does not implement stop
            pass
        self.listener.join()

    def on_press(self, key: "keyboard.Key"):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: "keyboard.Key"):
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self):
        return self.pressed_keys


class DroneRPMController:
    def __init__(self):
        self.base_rpm = DRONE_HOVER_RPM
        self.rotation_delta = DRONE_ROTATION_DELTA
        self.thrust_delta = DRONE_THRUST_DELTA
        self.rpms = [self.base_rpm] * 4

    def update(self, pressed_keys):
        # reset to hover
        self.rpms = [self.base_rpm] * 4

        # acceleration (spacebar) - all rotors spin faster
        if keyboard.Key.space in pressed_keys:
            self.base_rpm = np.clip(self.base_rpm + self.thrust_delta, 0, 25000)
            self.rpms = [self.base_rpm] * 4

        # deceleration (shift) - all rotors spin slower
        if keyboard.Key.shift in pressed_keys:
            self.base_rpm = np.clip(self.base_rpm - self.thrust_delta, 0, 25000)
            self.rpms = [self.base_rpm] * 4

        # forward - front rotors spin faster
        if keyboard.Key.up in pressed_keys:
            self.rpms[0] += self.rotation_delta  # front left
            self.rpms[1] += self.rotation_delta  # front right
            self.rpms[2] -= self.rotation_delta  # back left
            self.rpms[3] -= self.rotation_delta  # back right

        # backward - back rotors spin faster
        if keyboard.Key.down in pressed_keys:
            self.rpms[0] -= self.rotation_delta  # front left
            self.rpms[1] -= self.rotation_delta  # front right
            self.rpms[2] += self.rotation_delta  # back left
            self.rpms[3] += self.rotation_delta  # back right

        # left - left rotors spin faster
        if keyboard.Key.left in pressed_keys:
            self.rpms[0] -= self.rotation_delta  # front left
            self.rpms[2] -= self.rotation_delta  # back left
            self.rpms[1] += self.rotation_delta  # front right
            self.rpms[3] += self.rotation_delta  # back right

        # right - right rotors spin faster
        if keyboard.Key.right in pressed_keys:
            self.rpms[0] += self.rotation_delta  # front left
            self.rpms[2] += self.rotation_delta  # back left
            self.rpms[1] -= self.rotation_delta  # front right
            self.rpms[3] -= self.rotation_delta  # back right

        self.rpms = np.clip(self.rpms, 0, 25000)
        return self.rpms


def main():
    parser = argparse.ArgumentParser(description="Genesis LiDAR/Depth Camera Visualization with Drone Teleop")
    parser.add_argument("-B", "--n_envs", type=int, default=0, help="Number of environments to replicate")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    parser.add_argument(
        "--pattern", type=str, default="spherical", choices=("spherical", "depth", "grid"), help="Sensor pattern type"
    )
    args = parser.parse_args()

    if IS_PYNPUT_AVAILABLE:
        kb = KeyboardDevice()
        kb.start()
    else:
        print("Keyboard teleop is disabled since pynput is not installed. To install, run `pip install pynput`.")

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-3.0, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.5),
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())

    # create ring of obstacles to visualize lidar sensor hits
    for i in range(NUM_CYLINDERS):
        angle = 2 * np.pi * i / NUM_CYLINDERS
        x = CYLINDER_RING_RADIUS * np.cos(angle)
        y = CYLINDER_RING_RADIUS * np.sin(angle)
        scene.add_entity(
            gs.morphs.Cylinder(
                height=1.5,
                radius=0.3,
                pos=(x, y, 0.75),
                fixed=True,
            )
        )

    for i in range(NUM_BOXES):
        angle = 2 * np.pi * i / NUM_BOXES + np.pi / 6
        x = BOX_RING_RADIUS * np.cos(angle)
        y = BOX_RING_RADIUS * np.sin(angle)
        scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.5, 2.0 * (i + 1) / NUM_BOXES),
                pos=(x, y, 1.0),
                fixed=False,
            )
        )

    # create drone
    drone = scene.add_entity(
        gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0.0, 0.5),
        )
    )

    sensor_kwargs = dict(
        entity_idx=drone.idx,
        pos_offset=(0.0, 0.0, 0.1),  # offset 0.1m above drone center to avoid drone geometry
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
        draw_debug=True,
    )

    if args.pattern == "depth":
        sensor = scene.add_sensor(gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(), **sensor_kwargs))
        scene.start_recording(
            data_func=(lambda: sensor.read_image()[0]) if args.n_envs > 0 else sensor.read_image,
            rec_options=gs.recorders.MPLImagePlot(),
        )
    else:
        if args.pattern == "grid":
            pattern_cfg = gs.sensors.GridPattern()
        else:
            if args.pattern != "spherical":
                gs.logger.warning(f"Unrecognized raycaster pattern: {args.pattern}. Using 'spherical' instead.")
            pattern_cfg = gs.sensors.SphericalPattern()

        sensor = scene.add_sensor(gs.sensors.Lidar(pattern=pattern_cfg, **sensor_kwargs))

    scene.build(n_envs=args.n_envs)

    if IS_PYNPUT_AVAILABLE:
        # Print control instructions
        print("\nDrone Keyboard Controls:")
        print("[↑/↓/←/→]: Move Forward/Backward/Left/Right")
        print("[space]: Increase Thrust (All RPMs up)")
        print("[shift]: Decrease Thrust (All RPMs down)")
        print("[\\]: Reset to hover")
        print("[esc]: Quit")
        print(f"\nHover RPM: {DRONE_HOVER_RPM:.1f}")

    drone_controller = DroneRPMController()

    def apply_rpm_to_all_envs(rpms: np.ndarray):
        if args.n_envs > 0:
            rpms_expanded = np.expand_dims(rpms, axis=0).repeat(args.n_envs, axis=0)
        else:
            rpms_expanded = rpms
        drone.set_propellels_rpm(rpms_expanded)

    try:
        while True:
            if IS_PYNPUT_AVAILABLE:
                pressed = kb.pressed_keys.copy()
                if keyboard.Key.esc in pressed:
                    break
                if keyboard.KeyCode.from_char("\\") in pressed:
                    drone_controller.base_rpm = DRONE_HOVER_RPM
                    drone_controller.rpms = [DRONE_HOVER_RPM] * 4

                rpms = drone_controller.update(pressed)
            else:
                # hover in place if no keyboard control
                rpms = [DRONE_HOVER_RPM] * 4

            apply_rpm_to_all_envs(np.array(rpms, dtype=np.float32))
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")
        if IS_PYNPUT_AVAILABLE:
            kb.stop()


if __name__ == "__main__":
    main()
