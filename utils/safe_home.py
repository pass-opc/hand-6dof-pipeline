"""
Safety utility: record arm position on connect, return to it on disconnect.

Prevents the arm from falling when servos lose torque after disconnect.
Use as a context manager wrapping any RobotArm operation.

Usage:
    from robots.so101 import SO101Arm
    from utils.safe_home import SafeHome

    robot = SO101Arm(port="COM5")
    robot.connect()
    with SafeHome(robot) as home:
        # ... do stuff with robot ...
    # Arm automatically returns to home position before disconnect.
"""

import time

from robots.base import RobotArm


class SafeHome:
    """Context manager that records home position and restores it on exit.

    On __enter__: reads current joint + gripper positions as "home".
    On __exit__:  slowly moves back to home, then disconnects.

    Works with any RobotArm implementation.
    """

    def __init__(
        self,
        robot: RobotArm,
        return_duration_s: float = 3.0,
        step_hz: float = 30.0,
    ):
        self.robot = robot
        self.return_duration_s = return_duration_s
        self.step_hz = step_hz
        self.home_joints: dict[str, float] | None = None
        self.home_gripper: float | None = None

    def __enter__(self):
        self._record_home()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.return_home()
        except Exception as e:
            print(f"  [SafeHome] WARNING: failed to return home: {e}")
        finally:
            try:
                self.robot.disconnect()
                print(f"  [SafeHome] Robot disconnected.")
            except Exception as e:
                print(f"  [SafeHome] WARNING: disconnect failed: {e}")
        return False

    def _record_home(self):
        """Read and store current joint + gripper positions."""
        self.home_joints = self.robot.get_joint_positions()
        self.home_gripper = self.robot.get_gripper_position()

        pos_str = ", ".join(
            f"{name}: {val:.1f}°"
            for name, val in self.home_joints.items()
        )
        pos_str += f", gripper: {self.home_gripper:.1f}°"
        print(f"  [SafeHome] Home recorded: {pos_str}")

    def return_home(self):
        """Slowly interpolate from current position back to recorded home."""
        if self.home_joints is None:
            print(f"  [SafeHome] No home position recorded, skipping.")
            return

        current_joints = self.robot.get_joint_positions()
        current_gripper = self.robot.get_gripper_position()

        max_dist = max(
            abs(current_joints[k] - self.home_joints[k])
            for k in self.home_joints
        )
        max_dist = max(max_dist, abs(current_gripper - self.home_gripper))
        print(f"\n  [SafeHome] Returning to home position "
              f"({self.return_duration_s}s, max distance {max_dist:.1f}°)...")

        n_steps = int(self.return_duration_s * self.step_hz)
        dt = 1.0 / self.step_hz

        for s in range(1, n_steps + 1):
            t0 = time.perf_counter()
            alpha = s / n_steps

            interp_joints = {
                k: current_joints[k] + alpha * (self.home_joints[k] - current_joints[k])
                for k in self.home_joints
            }
            interp_gripper = current_gripper + alpha * (self.home_gripper - current_gripper)

            self.robot.send_all_positions(interp_joints, interp_gripper)

            elapsed = time.perf_counter() - t0
            time.sleep(max(dt - elapsed, 0))

        print(f"  [SafeHome] Home position reached.")
