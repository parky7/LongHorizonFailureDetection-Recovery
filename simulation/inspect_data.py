"""Validate recorded episodes without importing the simulator or using a GPU."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SKILLS = ("approach_object", "grasp_object", "lift_object", "move_to_target", "release_object")


def validate_episode(path):
    errors = []
    with h5py.File(path, "r") as file:
        success = bool(file.attrs.get("success", False))
        if file.attrs.get("status") != "complete":
            errors.append("Episode did not finish cleanly")
        if "error" in file.attrs:
            errors.append(str(file.attrs["error"]))
        steps = file.get("steps", {})
        count = len(steps)
        if not count:
            errors.append("No control steps")
        if list(steps) != [f"{i:06d}" for i in range(count)]:
            errors.append("Noncontiguous control steps")
        cameras = {}
        for i, step in enumerate(steps.values()):
            if not np.isclose(step.attrs["simulation_time_s"], (i + 1) / file.attrs["control_freq"]):
                errors.append(f"Incorrect timestamp at step {i}")
            for key in ("action", "oracle/qpos", "oracle/qvel", "oracle/cube_pose", "oracle/tcp_pose"):
                if not np.isfinite(step[key][()]).all():
                    errors.append(f"Nonfinite {key} at step {i}")
            images_by_camera = step.get("observations", {})
            if not images_by_camera:
                errors.append(f"Missing camera observations at step {i}")
            for camera, images in images_by_camera.items():
                rgb, depth = images["rgb"], images["depth"]
                if rgb.shape != depth.shape[:2] + (3,) or depth.shape[-1:] != (1,):
                    errors.append(f"Invalid RGB-D shapes at step {i}")
                if rgb.dtype != np.uint8 or depth.dtype != np.int16:
                    errors.append(f"Unexpected native RGB-D dtypes at step {i}: {rgb.dtype}, {depth.dtype}")
                if i == 0:
                    if not np.any(depth[()] > 0):
                        errors.append("Depth has no positive measurements")
                    cameras[camera] = {"rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
                                       "rgb_dtype": str(rgb.dtype), "depth_dtype": str(depth.dtype)}
        boundaries = file.get("skills", {})
        ordered = sorted(boundaries.values(), key=lambda b: b.attrs["start_step"])
        previous_end = 0
        for boundary in ordered:
            start, end = int(boundary.attrs["start_step"]), int(boundary.attrs["end_step_exclusive"])
            if start != previous_end or end < start or end > count:
                errors.append(f"Invalid boundary {boundary.name}")
            name = boundary.attrs.get("name", boundary.name.rsplit("/", 1)[-1])
            for i in range(start, end):
                if steps[f"{i:06d}"].attrs["skill"] != name:
                    errors.append(f"Skill boundary disagrees with step {i}")
            previous_end = end
        if previous_end != count:
            errors.append("Skill boundaries do not cover all steps")
        names = [b.attrs.get("name", b.name.rsplit("/", 1)[-1]) for b in ordered]
        if success and (not ordered or names[-1] != "release_object"
                        or not ordered[-1].attrs["success"] or not set(SKILLS).issubset(names)):
            errors.append("Success label lacks a completed placement sequence")
        if success and count:
            last = steps[f"{count - 1:06d}"]["oracle"]
            if bool(last["is_grasped"][()]) or not bool(last["success"][()]):
                errors.append("Success label disagrees with final simulator state")
        return {"path": str(path), "seed": int(file.attrs["seed"]), "success": success,
                "steps": count, "cameras": cameras, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-successes", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = []
    for path in sorted(args.directory.glob("episode_*.h5")):
        try:
            result = validate_episode(path)
        except (KeyError, ValueError, OSError) as error:
            result = {"path": str(path), "success": False, "errors": [str(error)]}
        results.append(result)
    successes = sum(r["success"] and not r["errors"] for r in results)
    report = {"episodes": len(results), "valid_successes": successes, "results": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not results or any(r["errors"] for r in results) or successes < args.require_successes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
