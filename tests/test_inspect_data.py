import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from simulation.inspect_data import SKILLS, validate_episode


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "episode_000000.h5"
        with h5py.File(self.path, "w") as file:
            file.attrs.update(seed=0, status="complete", success=True, control_freq=20)
            for i, skill in enumerate(SKILLS):
                step = file.create_group(f"steps/{i:06d}")
                step.attrs.update(skill=skill, simulation_time_s=(i + 1) / 20)
                step.create_dataset("action", data=np.zeros(8))
                for key in ("qpos", "qvel", "cube_pose", "tcp_pose"):
                    step.create_dataset("oracle/" + key, data=np.zeros(9))
                step.create_dataset("oracle/is_grasped", data=False)
                step.create_dataset("oracle/success", data=True)
                step.create_dataset("observations/camera/rgb", data=np.zeros((4, 4, 3), np.uint8))
                step.create_dataset("observations/camera/depth", data=np.ones((4, 4, 1), np.int16))
                boundary = file.create_group("skills/" + skill)
                boundary.attrs.update(start_step=i, end_step_exclusive=i + 1, success=True)

    def test_complete_episode(self):
        self.assertEqual(validate_episode(self.path)["errors"], [])

    def test_incomplete_file_cannot_count_as_success(self):
        with h5py.File(self.path, "a") as file:
            file.attrs["status"] = "running"
        self.assertIn("Episode did not finish cleanly", validate_episode(self.path)["errors"])

    def test_reject_incorrect_final_success(self):
        with h5py.File(self.path, "a") as file:
            file["steps/000004/oracle/is_grasped"][()] = True
        self.assertIn("Success label disagrees with final simulator state", validate_episode(self.path)["errors"])

    def test_reject_misaligned_boundary(self):
        with h5py.File(self.path, "a") as file:
            file["skills/grasp_object"].attrs["start_step"] = 0
        self.assertTrue(validate_episode(self.path)["errors"])


if __name__ == "__main__":
    unittest.main()
