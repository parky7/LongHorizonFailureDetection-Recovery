"""Single-environment Panda skills adapted from ManiSkill's PickCube expert."""
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb


class MotionPlanningFailure(RuntimeError):
    """A valid planning request found no path; the live episode may recover."""


class PandaSkills:
    def __init__(self, env):
        self.env = env.unwrapped
        self.planner = PandaArmMotionPlanningSolver(
            env, base_pose=self.env.agent.robot.pose, vis=False,
            visualize_target_grasp_pose=False, print_env_info=False,
        )
        self.grasp_pose = None

    def move(self, pose):
        result = self.planner.move_to_pose_with_screw(pose)
        if isinstance(result, (int, np.integer)) and result == -1:
            raise MotionPlanningFailure("Motion planner could not find a path")

    def approach_object(self):
        # Recompute from the actual scene each time, including after a disturbance.
        self.planner.open_gripper()
        closing = self.env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp = compute_grasp_info_by_obb(
            get_actor_obb(self.env.cube), approaching=np.array([0, 0, -1]),
            target_closing=closing, depth=0.025,
        )
        self.grasp_pose = self.env.agent.build_grasp_pose(
            np.array([0, 0, -1]), grasp["closing"], self.env.cube.pose.sp.p,
        )
        self.move(self.grasp_pose * sapien.Pose([0, 0, -0.05]))

    def grasp_object(self):
        if self.grasp_pose is None:
            raise RuntimeError("approach_object must precede grasp_object")
        self.move(self.grasp_pose)
        self.planner.close_gripper(t=12)

    def lift_object(self):
        pose = self.env.agent.tcp.pose.sp
        self.move(sapien.Pose(pose.p + np.array([0, 0, 0.10]), pose.q))

    def move_to_target(self):
        # Preserve the measured object-to-TCP offset rather than assuming zero.
        offset = self.env.agent.tcp.pose.sp.p - self.env.cube.pose.sp.p
        target = self.env.goal_site.pose.sp.p + offset
        self.move(sapien.Pose(target + np.array([0, 0, 0.10]), self.grasp_pose.q))
        self.move(sapien.Pose(target, self.grasp_pose.q))
        self.planner.close_gripper(t=12)

    def release_object(self):
        self.planner.open_gripper(t=12)
        pose = self.env.agent.tcp.pose.sp
        self.move(sapien.Pose(pose.p + np.array([0, 0, 0.08]), pose.q))
        self.planner.open_gripper(t=24)
