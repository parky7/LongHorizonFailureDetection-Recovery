"""Controlled simulator interventions; onset and outcome are logged separately."""
import numpy as np
import sapien


def drop_object(env, planner, on_displaced=None, offset=(0.08, 0.04)):
    """Displace a held cube sideways, then let gravity settle it on the table.

    This is an explicit pose intervention, not a realistic slip dynamics model.
    Keep the robot at its current pose while the cube falls. The caller checks
    the resulting state rather than treating injection as a failure label.
    """
    base = env.unwrapped
    pose = base.cube.pose.sp
    base.cube.set_pose(sapien.Pose(pose.p + np.array([*offset, 0]), pose.q))
    base.cube.set_linear_velocity(np.zeros(3))
    base.cube.set_angular_velocity(np.zeros(3))
    if on_displaced is not None:
        on_displaced()
    planner.close_gripper(t=16)
