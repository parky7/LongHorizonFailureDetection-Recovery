"""Evaluate grasp-loss detection on completed holding skills and settled drops."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py

from detectors.aperture import ApertureMonitor, HOLDING_SKILLS


def evaluate(directory, monitor):
    counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    episodes, samples = 0, 0
    for path in sorted(directory.rglob("episode_*.h5")):
        with h5py.File(path) as file:
            if file.attrs.get("status") != "complete" or "error" in file.attrs:
                raise ValueError(f"Incomplete or errored recording: {path}")
            episodes += 1
            for boundary in file["skills"].values():
                skill = boundary.attrs.get("name", boundary.name.rsplit("/", 1)[-1])
                # Exclude skill attempts where a path failed before any execution.
                start, end = boundary.attrs["start_step"], boundary.attrs["end_step_exclusive"]
                if skill not in HOLDING_SKILLS or end <= start:
                    continue
                state = file[f"steps/{end - 1:06d}/oracle"]
                prediction = monitor.grasp_lost(skill, state["qpos"][()])
                # Oracle is used only to score predictions, never as an input.
                truth = not bool(state["is_grasped"][()])
                key = ("true_" if prediction == truth else "false_") + ("positive" if prediction else "negative")
                counts[key] += 1
                samples += 1
    tp, fp, fn = counts["true_positive"], counts["false_positive"], counts["false_negative"]
    return {"episodes": episodes, "boundary_samples": samples, **counts,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    monitor = ApertureMonitor()
    result = {"minimum_aperture_m": monitor.minimum_aperture_m,
              "coverage": "grasp loss at completed holding-skill boundaries; not full task failure detection",
              "unit": "boundary samples; multiple samples share an episode; not event-level F1",
              "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])):
                                hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "detectors/aperture.py")},
              "conditions": {p.name: evaluate(p, monitor) for p in sorted(args.directory.iterdir())
                             if p.is_dir() and any(p.glob("episode_*.h5"))}}
    if not result["conditions"]:
        raise SystemExit("No completed episode directories found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
