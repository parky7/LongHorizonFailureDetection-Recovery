import unittest
import tempfile
from pathlib import Path
import h5py
import numpy as np

from detectors.aperture import ApertureMonitor
from evaluation.aperture import evaluate


class ApertureTests(unittest.TestCase):
    def setUp(self):
        self.monitor = ApertureMonitor()
        self.qpos = np.zeros(9)

    def test_empty_closed_fingers_indicate_grasp_loss(self):
        self.assertTrue(self.monitor.grasp_lost("lift_object", self.qpos))

    def test_fingers_separated_by_cube_do_not_indicate_loss(self):
        self.qpos[-2:] = .02
        self.assertFalse(self.monitor.grasp_lost("grasp_object", self.qpos))

    def test_release_has_no_holding_requirement(self):
        self.assertIsNone(self.monitor.grasp_lost("release_object", self.qpos))

    def test_bad_robot_state_is_rejected(self):
        with self.assertRaises(ValueError):
            self.monitor.grasp_lost("lift_object", np.zeros(7))
        self.qpos[0] = np.nan
        with self.assertRaises(ValueError):
            self.monitor.grasp_lost("lift_object", self.qpos)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            ApertureMonitor(-.1)

    def test_offline_scoring_keeps_oracle_out_of_prediction_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "episode_000000.h5", "w") as file:
                file.attrs["status"] = "complete"
                # (measured aperture, oracle grasp): TN, TP, FP, FN.
                for i, (aperture, grasped) in enumerate(((.04, True), (0., False), (0., True), (.04, False))):
                    boundary = file.create_group(f"skills/{i:04d}")
                    boundary.attrs.update(name="lift_object", start_step=i, end_step_exclusive=i + 1)
                    qpos = np.zeros(9)
                    qpos[-2:] = aperture / 2
                    file.create_dataset(f"steps/{i:06d}/oracle/qpos", data=qpos)
                    file.create_dataset(f"steps/{i:06d}/oracle/is_grasped", data=grasped)
            result = evaluate(Path(directory), self.monitor)
        for key in ("true_positive", "true_negative", "false_positive", "false_negative"):
            self.assertEqual(result[key], 1)
        self.assertEqual(result["f1"], .5)


if __name__ == "__main__":
    unittest.main()
