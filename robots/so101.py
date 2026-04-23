"""
SO-arm 101 driver — wraps LeRobot's SO100Follower for the RobotArm interface.

Hardware:
    5 revolute joints (shoulder_pan → wrist_roll) + 1 gripper
    Feetech STS3215 servos, serial bus via USB-to-TTL adapter

Requires:
    - LeRobot installed (lerobot.robots.so_follower)
    - Calibration file at ~/.cache/huggingface/lerobot/calibration/robots/so_follower/{id}.json
    - URDF at assets/so101_new_calib.urdf (relative to repo root)
"""

from pathlib import Path

from robots.base import RobotArm

# Project root — two levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default URDF location (relative to project root)
_DEFAULT_URDF = _PROJECT_ROOT / "assets" / "so101_new_calib.urdf"

# Joint names in URDF / motor bus order (excluding gripper)
_JOINT_NAMES = [
    "shoulder_pan",   # axis 1: base rotation
    "shoulder_lift",  # axis 2: upper arm pitch
    "elbow_flex",     # axis 3: elbow bend
    "wrist_flex",     # axis 4: wrist pitch
    "wrist_roll",     # axis 5: wrist rotation
]


class SO101Arm(RobotArm):
    """SO-arm 101 driver using LeRobot SO100Follower backend.

    Args:
        port: serial port (e.g. "COM5", "/dev/ttyUSB0")
        id: calibration file name (must match file in
            ~/.cache/huggingface/lerobot/calibration/robots/so_follower/)
        max_relative_target: max degrees the arm can move per command step.
            Lower = safer but slower tracking.
        urdf_path: override URDF location
    """

    def __init__(
        self,
        port: str,
        id: str = "pass_follower_arm",
        max_relative_target: float = 10.0,
        urdf_path: Path | None = None,
    ):
        self._port = port
        self._id = id
        self._max_relative_target = max_relative_target
        self._urdf = urdf_path or _DEFAULT_URDF
        self._robot = None  # SO100Follower instance, created on connect()

    # --- Properties ---

    @property
    def joint_names(self) -> list[str]:
        return _JOINT_NAMES

    @property
    def urdf_path(self) -> Path:
        return self._urdf

    # --- Connection ---

    def connect(self) -> None:
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

        config = SO100FollowerConfig(
            port=self._port,
            use_degrees=True,
            max_relative_target=self._max_relative_target,
            id=self._id,
        )
        self._robot = SO100Follower(config)
        self._robot.connect()
        print(f"  [SO101] Connected on {self._port} (id={self._id})")

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
            print(f"  [SO101] Disconnected.")

    # --- Arm joints ---

    def get_joint_positions(self) -> dict[str, float]:
        obs = self._robot.get_observation()
        return {name: obs[f"{name}.pos"] for name in _JOINT_NAMES}

    def send_joint_positions(self, positions: dict[str, float]) -> None:
        action = {f"{name}.pos": positions[name] for name in positions}
        self._robot.send_action(action)

    # --- Gripper ---

    def get_gripper_position(self) -> float:
        obs = self._robot.get_observation()
        return obs["gripper.pos"]

    def send_gripper_position(self, degrees: float) -> None:
        self._robot.send_action({"gripper.pos": degrees})
