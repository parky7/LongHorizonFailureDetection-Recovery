import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py

from simulation.campaign import main


class CampaignTests(unittest.TestCase):
    def run_campaign(self, paired=True, invalid=False, collision=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        test_manifest = root / "test.json"
        test_manifest.write_text(json.dumps({"episodes": [
            {"split": "test", "seed": 2_000_000 if collision else 1000,
             "success": True, "path": "reserved"}]}))
        output = root / "run"
        clock = [100.0]

        def benchmark(command, **kwargs):
            directory = Path(command[command.index("--output") + 1])
            directory.mkdir()
            seed = int(command[command.index("--seed") + 1])
            if not invalid:
                results = {}
                for condition in ("nominal", "drop_fixed", "drop_retry", "drop_scripted"):
                    condition_path = directory / condition
                    condition_path.mkdir()
                    with h5py.File(condition_path / f"episode_{seed:06d}.h5", "w") as file:
                        file.attrs.update(seed=seed, status="complete", success=True, task="fixture")
                        file.create_group("steps/000000")
                    results[condition] = {"episodes": 1}
                (directory / "results.json").write_text(json.dumps(results))
                if paired:
                    (directory / "report.json").write_text("{}")

            class Process:
                def wait(self, timeout):
                    clock[0] = 9950 if not invalid else clock[0] + 1
                    return 1 if invalid or not paired else 0
            return Process()

        arguments = ["campaign", "--output", str(output), "--test-manifest", str(test_manifest),
                     "--end-time", "10000", "--batch-seeds", "1"]
        with patch("sys.argv", arguments), patch("simulation.campaign.time.time", side_effect=lambda: clock[0]), \
                patch("simulation.campaign.subprocess.Popen", side_effect=benchmark), \
                patch("simulation.campaign.signal.signal"), patch("builtins.print"):
            if invalid or collision:
                with self.assertRaises(RuntimeError):
                    main()
            else:
                main()
        return json.loads((output / "status.json").read_text()), output

    def test_valid_batch_is_checkpointed_before_deadline(self):
        status, output = self.run_campaign()
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["stop_reason"], "allocation_deadline")
        self.assertEqual(status["summary"]["validation"]["episodes"], 4)
        self.assertTrue(status["batches"][0]["matched_states_verified"])
        manifest = json.loads((output / "dataset_manifest.json").read_text())
        self.assertEqual({r["seed"] for r in manifest["episodes"]}, {1000, 2_000_000})

    def test_valid_unpaired_collection_is_labeled_and_retained(self):
        status, _ = self.run_campaign(paired=False)
        self.assertEqual(status["summary"]["validation"]["episodes"], 4)
        self.assertFalse(status["batches"][0]["matched_states_verified"])

    def test_repeated_invalid_batches_stop_and_stay_out_of_manifest(self):
        status, _ = self.run_campaign(invalid=True)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(len(status["excluded_batches"]), 3)
        self.assertEqual(status["summary"]["train"]["episodes"], 0)
        self.assertEqual(status["summary"]["validation"]["episodes"], 0)

    def test_reserved_test_seeds_cannot_be_used_for_collection(self):
        status, _ = self.run_campaign(collision=True)
        self.assertEqual(status["state"], "failed")
        self.assertIn("overlap", status["error"])


if __name__ == "__main__":
    unittest.main()
