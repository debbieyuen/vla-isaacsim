# Collects the simulated Unitree H1's robot state and actions.
# We store them each timestep and save them to a file for future training.

from __future__ import annotations
from typing import Any, Optional
import numpy as np

class DataLogs:
    def __init__(self):
        self.data = []

    def log(
        self,
        robot: Any,
        controller: Any,
        step: Optional[int] = None,
        behavior: Optional[str] = None,
    ):
        position, _ = robot.get_world_pose()
        joint_names = [str(n) for n in list(robot.dof_names)]
        joint_positions = robot.get_joint_positions().copy()
        joint_velocities = robot.get_joint_velocities().copy()

        joint_state = {
            name: {
                "pos": float(pos),
                "vel": float(vel),
            }
            for name, pos, vel in zip(joint_names, joint_positions, joint_velocities)
        }

        # controller is expected to be the policy wrapper object, which has .action
        action = None
        if hasattr(controller, "action"):
            action = controller.action.copy()
        else:
            # fallback to pass an action array directly
            action = np.asarray(controller).copy()

        self.data.append(
            {
                "step": step,
                "behavior": behavior,
                "joint_names": joint_names,
                "joint_positions": joint_positions,
                "joint_velocities": joint_velocities,
                "joint_state": joint_state,
                "action": action,
                "base_position": position.copy(),
                "base_velocity": robot.get_linear_velocity().copy(),
                "base_angular_velocity": robot.get_angular_velocity().copy(),
            }
        )

    def save(self, path="rollout.npy"):
        np.save(path, self.data)