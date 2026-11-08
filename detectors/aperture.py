"""A Panda grasp-loss baseline using joint readings, evaluated at skill boundaries.

The 20 mm aperture threshold is half the known 40 mm cube width. It is a
geometry-based baseline, not a learned or general object-grasp detector.
It cannot verify lift height, placement accuracy, or grasp contact forces.
"""
import numpy as np

HOLDING_SKILLS = {"grasp_object", "lift_object", "move_to_target"}


class ApertureMonitor:
    def __init__(self, minimum_aperture_m=0.02):
        if not np.isfinite(minimum_aperture_m) or minimum_aperture_m <= 0:
            raise ValueError("minimum_aperture_m must be finite and positive")
        self.minimum_aperture_m = minimum_aperture_m

    def grasp_lost(self, skill, qpos):
        """Return True/False for holding postconditions, None for other skills.

        Call after a skill completes or while a previously verified grasp is
        expected to persist. Do not apply while the fingers are still closing.
        Panda qpos has seven arm joints and two finger positions, in meters.
        No cube pose, contact/grasp flag, image, disturbance metadata or outcome
        is accepted as an input.
        """
        qpos = np.asarray(qpos)
        if qpos.shape != (9,) or not np.isfinite(qpos).all():
            raise ValueError("Expected nine finite Panda joint positions")
        if skill not in HOLDING_SKILLS:
            return None
        return bool(qpos[-2:].sum() < self.minimum_aperture_m)
