"""Run matched nominal, fixed, retry and scripted-recovery conditions sequentially."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from simulation.inspect_data import validate_episode

CONDITIONS = {
    "nominal": ("none", "stop"),
    "drop_fixed": ("drop_after_lift", "continue"),
    "drop_retry": ("drop_after_lift", "retry"),
    "drop_scripted": ("drop_after_lift", "scripted"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-budget", type=float, default=1800)
    parser.add_argument("--drop-offset", type=float, nargs=2, default=(0.08, 0.04))
    args = parser.parse_args()
    if args.episodes < 1 or args.time_budget <= 0:
        parser.error("--episodes and --time-budget must be positive")
    # A fresh run directory keeps conditions and summaries from different runs apart.
    import math
    if not all(math.isfinite(v) and abs(v) <= 0.2 for v in args.drop_offset):
        parser.error("--drop-offset components must be finite and within +/-0.2 meters")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    source_files = list((root / "simulation").glob("*.py")) + list(root.glob("requirements*.txt"))
    for path in sorted(source_files):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = args.output / "source" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    config = {"episodes": args.episodes, "seed": args.seed, "time_budget_s": args.time_budget,
              "conditions": CONDITIONS, "drop_offset_m": args.drop_offset,
              "source_sha256": hashes, "python": sys.executable,
              "python_version": sys.version,
              "slurm_job_id": os.getenv("SLURM_JOB_ID"), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.monotonic()
    results = {}
    for condition, (disturbance, recovery) in CONDITIONS.items():
        remaining = args.time_budget - (time.monotonic() - started)
        if remaining <= 0:
            raise SystemExit("Benchmark time budget exhausted")
        directory = args.output / condition
        command = [sys.executable, "-u", "-m", "simulation.collect", "--episodes", str(args.episodes),
                   "--seed", str(args.seed), "--output", str(directory), "--disturbance", disturbance,
                   "--recovery", recovery, "--max-retries", "2", "--time-budget", str(remaining),
                   "--drop-offset", *map(str, args.drop_offset)]
        print(f"Starting {condition}; log: {args.output / (condition + '.log')}", flush=True)
        with (args.output / (condition + ".log")).open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=remaining + 30, cwd=root,
                                     env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        episodes = [validate_episode(p) for p in sorted(directory.glob("episode_*.h5"))]
        if len(episodes) != args.episodes or any(r["errors"] for r in episodes):
            raise RuntimeError(f"Incomplete or invalid {condition} run; inspect its log and HDF5 files")
        # Exit 1 is expected when a valid run contains unsuccessful tasks.
        if process.returncode not in (0, 1):
            raise RuntimeError(f"{condition} exited with code {process.returncode}")
        summary = json.loads((directory / "summary.json").read_text())
        results[condition] = {
            "episodes": len(episodes), "successes": sum(r["success"] for r in episodes),
            "mean_steps": float(np.mean([r["steps"] for r in summary])),
            "mean_wall_time_s": float(np.mean([r["wall_time_s"] for r in summary])),
            "episodes_with_detection": sum(r["failures_detected"] > 0 for r in summary),
            "mean_recovery_attempts": float(np.mean([r["recovery_attempts"] for r in summary]))}
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{condition}: {results[condition]}", flush=True)

    # Verify paired initial states and identical pre-intervention states.
    for seed in range(args.seed, args.seed + args.episodes):
        filename = f"episode_{seed:06d}.h5"
        with h5py.File(args.output / "nominal" / filename) as reference:
            initial = {k: v[()] for k, v in reference["initial/oracle"].items()}
        before_reference = None
        for condition in CONDITIONS:
            with h5py.File(args.output / condition / filename) as file:
                for key, value in initial.items():
                    if not np.allclose(value, file["initial/oracle/" + key][()], atol=1e-6, rtol=0):
                        raise RuntimeError(f"Unmatched initial state for seed={seed} {condition}: {key}")
                if condition == "nominal":
                    continue
                disturbances = [g for g in file.get("events", {}).values() if g.attrs["kind"] == "disturbance"]
                if len(disturbances) != 1:
                    raise RuntimeError(f"Expected one intervention for seed={seed} {condition}")
                before = {k: v[()] for k, v in disturbances[0]["before"].items()}
                if before_reference is None:
                    before_reference = before
                elif any(not np.allclose(v, before_reference[k], atol=1e-6, rtol=0) for k, v in before.items()):
                    raise RuntimeError(f"Unmatched intervention state for seed={seed} {condition}")
    report = {"conditions": results, "matched_states_verified": True,
              "wall_time_s": time.monotonic() - started}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
