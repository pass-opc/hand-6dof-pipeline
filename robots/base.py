"""
Abstract robot arm interface for trajectory replay.

Any robot driver must implement RobotArm to be used with
scripts/04_replay_on_arm.py.

Input:  joint names, URDF path, serial port
Output: connect/disconnect, read/write joint positions and gripper
"""

from abc import ABC, abstractmethod
from pathlib import Path


class RobotArm(ABC):
    """Abstract interface for a robot arm with gripper."""

    # --- Properties that subclasses must define ---

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        """Ordered list of arm joint names (excluding gripper)."""
        ...

    @property
    @abstractmethod
    def urdf_path(self) -> Path:
        """Path to URDF file for IK chain construction."""
        ...

    @property
    def n_joints(self) -> int:
        """Number of arm joints (excluding gripper)."""
        return len(self.joint_names)

    # --- Connection ---

    @abstractmethod
    def connect(self) -> None:
        """Open serial/hardware connection."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and release resources."""
        ...

    # --- Arm joint read/write (degrees) ---

    @abstractmethod
    def get_joint_positions(self) -> dict[str, float]:
        """Read current joint angles. Returns {joint_name: degrees}."""
        ...

    @abstractmethod
    def send_joint_positions(self, positions: dict[str, float]) -> None:
        """Command joint angles. Input: {joint_name: degrees}.

        The driver is responsible for applying safety limits
        (e.g., max_relative_target clamping).
        """
        ...

    # --- Gripper read/write (degrees) ---

    @abstractmethod
    def get_gripper_position(self) -> float:
        """Read current gripper angle in degrees."""
        ...

    @abstractmethod
    def send_gripper_position(self, degrees: float) -> None:
        """Command gripper angle in degrees."""
        ...

    # --- Convenience ---

    def get_all_positions(self) -> dict[str, float]:
        """Read all positions (arm joints + gripper)."""
        pos = self.get_joint_positions()
        pos["gripper"] = self.get_gripper_position()
        return pos

    def send_all_positions(
        self, joint_pos: dict[str, float], gripper_deg: float
    ) -> None:
        """Command all positions (arm joints + gripper)."""
        self.send_joint_positions(joint_pos)
        self.send_gripper_position(gripper_deg)
