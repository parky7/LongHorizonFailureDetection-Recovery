import tempfile
import unittest
from pathlib import Path

import h5py

from data.manifest import build_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def episode(self, name, seed, success=True, status="complete"):
        path = self.root / name / f"episode_{seed:06d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as file:
            file.attrs.update(seed=seed, status=status, success=success,
                              task="PickCube-tabletop-placement-v0")
            file.create_group("steps/000000")
        return path

    def test_same_seed_conditions_can_share_a_split(self):
        a = self.episode("nominal", 1)
        b = self.episode("drop", 1, success=False)
        result = build_manifest({"train": [a, b]})
        self.assertEqual(result["summary"]["train"],
                         {"episodes": 2, "unique_seeds": 1, "successes": 1, "failures": 1})

    def test_conditions_of_one_scene_cannot_cross_splits(self):
        a = self.episode("nominal", 1)
        b = self.episode("drop", 1, success=False)
        with self.assertRaisesRegex(ValueError, "leaks across"):
            build_manifest({"train": [a], "test": [b]})

    def test_independent_seeds_and_failed_tasks_are_retained(self):
        a = self.episode("train", 1)
        b = self.episode("test", 2, success=False)
        result = build_manifest({"train": [a], "test": [b]})
        self.assertEqual(result["summary"]["test"]["failures"], 1)

    def test_interrupted_recording_is_rejected(self):
        path = self.episode("interrupted", 1, status="running")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            build_manifest({"train": [path]})


if __name__ == "__main__":
    unittest.main()
