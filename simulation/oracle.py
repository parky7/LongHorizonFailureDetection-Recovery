"""Privileged postconditions, deliberately separate from learned inputs."""
import numpy as np


def check(skill, state, initial_height, goal_thresh=0.025):
    distance = float(np.linalg.norm(state["cube_pose"][:3] - state["goal_pose"][:3]))
    if skill == "approach_object":
        return bool(np.linalg.norm(state["tcp_pose"][:3] - state["cube_pose"][:3]) < 0.09)
    if skill == "grasp_object":
        return bool(state["is_grasped"])
    if skill == "lift_object":
        return bool(state["is_grasped"] and state["cube_pose"][2] > initial_height + 0.05)
    if skill == "move_to_target":
        return bool(state["is_grasped"] and distance <= goal_thresh)
    if skill == "release_object":
        return bool(not state["is_grasped"] and distance <= goal_thresh and state["success"])
    raise ValueError(f"Unknown skill: {skill}")
