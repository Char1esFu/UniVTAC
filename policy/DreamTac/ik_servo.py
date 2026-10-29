"""Cartesian -> joint servo for absolute end-effector actions.

Dream-Tac predicts absolute EE poses, while the evaluation loop drives the arm
through joint targets. This module turns one into the other with curobo's IK
solver, so `envs/` needs no changes: we hand `task.take_action(..., 'qpos')`
exactly what an ACT policy would.

Two choices here are deliberate and worth keeping:

* **A private IKSolver, not the planner's MotionGen.** `CuroboPlanner` carries a
  collision world rebuilt from every actor in the scene. During a contact-rich
  insertion the grasped object sits in that world, which makes the start state
  self-colliding and every solve fail unless the actor is registered in
  `ignore_actors`. A world-free solver sidesteps that, is much faster, and
  matches what the ACT baseline does (namely: no collision checking at all).
  Self-collision checking stays on -- it costs nothing and catches nonsense.

* **Seeding every solve with the current configuration.** The Panda is 7-DoF, so
  a 6-DoF pose leaves a one-dimensional null space: infinitely many joint
  configurations reach the same EE pose. An unseeded solve picks whichever
  branch its random start converges to, and consecutive steps can land in
  different branches -- the end-effector path stays continuous while the arm
  flips its elbow through half its workspace. Passing the current qpos as both
  `seed_config` (optimisation start) and `retract_config` (regularisation
  target) returns the solution nearest to where the arm already is. This is the
  recipe curobo's own `solve_single` docstring prescribes for IK servoing.
"""

from pathlib import Path

import numpy as np
import torch
import yaml


class IKServo:
    """Absolute EE pose (robot base frame) -> arm joint targets."""

    def __init__(
        self,
        robot_manager,
        num_seeds: int = 20,
        position_threshold: float = 0.001,
        rotation_threshold: float = 0.01,
        self_collision_check: bool = True,
        use_cuda_graph: bool = True,
    ):
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        self._robot = robot_manager
        self.device = robot_manager.device

        yaml_path = Path(robot_manager.yaml_path)
        with open(yaml_path, "r") as f:
            yml = yaml.safe_load(f)

        # CuroboPlanner resolves these relative to the yml; do the same so both
        # consumers load the identical kinematics.
        kinematics = yml["robot_cfg"]["kinematics"]
        for key in ("urdf_path", "collision_spheres"):
            value = kinematics.get(key)
            if value is not None and not Path(value).is_absolute():
                kinematics[key] = str(yaml_path.parent / value)

        # Applied by CuroboPlanner.plan_path to every goal; [0,0,0] for the
        # Franka, but read it rather than assume it.
        self.frame_bias = np.asarray(yml.get("planner", {}).get("frame_bias", [0.0, 0.0, 0.0]), dtype=np.float64)

        config = IKSolverConfig.load_from_robot_config(
            robot_cfg=yml,
            world_model=None,
            tensor_args=TensorDeviceType(device=torch.device(self.device)),
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_check,
            regularization=True,
            high_precision=True,
            use_cuda_graph=use_cuda_graph,
        )
        self.solver = IKSolver(config)
        self.dof = len(self._robot._arm_joint_names)

        self.num_solves = 0
        self.num_failures = 0

    def current_arm_qpos(self) -> torch.Tensor:
        """(dof,) float32, ordered like the solver's joint_names."""
        # _arm_ids is built from _arm_joint_names, which lists panda_joint1..7 in
        # the same order as the yml's cspace.joint_names.
        return self._robot.robot.data.joint_pos[0, self._robot._arm_ids].detach().float()

    def solve(self, position, quaternion_wxyz, seed_qpos=None):
        """Return arm joint targets reaching the pose, or None if unsolved.

        `position`/`quaternion_wxyz` are in the robot base (panda_link0) frame --
        the same frame `robot_manager.get_ee_pose()` reports and the same one the
        training data's `embodiment/ee` was recorded in, so no transform is
        needed beyond the planner's frame_bias.
        """
        from curobo.types.math import Pose as CuroboPose

        if seed_qpos is None:
            seed_qpos = self.current_arm_qpos()
        seed = seed_qpos.reshape(1, self.dof).to(self.device)

        position = np.asarray(position, dtype=np.float64) + self.frame_bias
        goal = CuroboPose(
            position=torch.tensor(position, dtype=torch.float32, device=self.device).reshape(1, 3),
            quaternion=torch.tensor(
                np.asarray(quaternion_wxyz, dtype=np.float64), dtype=torch.float32, device=self.device
            ).reshape(1, 4),
        )

        self.num_solves += 1
        result = self.solver.solve_single(
            goal,
            retract_config=seed,
            seed_config=seed.reshape(1, 1, self.dof),
        )
        if not bool(result.success.view(-1)[0].item()):
            self.num_failures += 1
            return None
        return result.solution.view(-1, self.dof)[0].detach().float()

    @property
    def failure_rate(self) -> float:
        return self.num_failures / max(self.num_solves, 1)
