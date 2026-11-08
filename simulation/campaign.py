"""Collect matched failure datasets in small batches until the allocation expires."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from data.manifest import build_manifest


OFFSETS = ((0.08, 0.04), (-0.08, 0.04), (0.04, -0.08),
           (-0.04, -0.08), (0.05, 0.025), (0.12, 0.06))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-time", type=float, required=True,
                        help="Unix time to stop, including shutdown margin")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--batch-seeds", type=int, default=10)
    args = parser.parse_args()
    if args.batch_seeds < 1 or args.end_time <= time.time():
        parser.error("Positive batch size and a future end time are required")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    if status_path.exists():
        parser.error("Use a fresh campaign directory")
    records = [r for r in json.loads(args.test_manifest.read_text())["episodes"]
               if r["split"] == "test"]
    if not records:
        parser.error("The test manifest must contain reserved test episodes")
    reserved = {r["seed"] for r in records}
    status = {"state": "running", "pid": os.getpid(),
              "slurm_job_id": os.getenv("SLURM_JOB_ID"),
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "stop_at_epoch": args.end_time, "batch_seeds": args.batch_seeds,
              "offsets_m": OFFSETS, "batches": [], "excluded_batches": []}
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Signal {signum}: stopping after the current batch", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def checkpoint():
        summary = {}
        for split in ("train", "validation", "test"):
            subset = [r for r in records if r["split"] == split]
            summary[split] = {"episodes": len(subset),
                              "unique_seeds": len({r["seed"] for r in subset}),
                              "successes": sum(r["success"] for r in subset),
                              "failures": sum(not r["success"] for r in subset)}
        status["summary"] = summary
        status["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(args.output / "dataset_manifest.json",
                    {"grouping": "complete seed across all conditions",
                     "summary": summary, "episodes": records})
        atomic_json(status_path, status)

    checkpoint()
    # Interleave validation from the start, using disjoint seed namespaces.
    counts = {"train": 0, "validation": 0}
    durations = []
    consecutive_errors = 0
    index = 0
    reason = "allocation_deadline"
    try:
        while not stop_requested:
            remaining = args.end_time - time.time()
            estimate = max(durations[-5:], default=240)
            if remaining < max(180, estimate * 1.4 + 30):
                break
            if shutil.disk_usage(args.output).free < 10 * 1024**3:
                reason = "less_than_10_GiB_free"
                break
            split = "validation" if index % 5 == 0 else "train"
            seed = (2_000_000 if split == "validation" else 20_000) + counts[split]
            seeds = set(range(seed, seed + args.batch_seeds))
            if seeds & reserved:
                raise RuntimeError("Campaign seeds overlap a reserved split")
            offset = OFFSETS[(index // 5 + index % 5) % len(OFFSETS)]
            directory = args.output / f"{index:05d}_{split}"
            command = [sys.executable, "-u", "-m", "simulation.benchmark",
                       "--episodes", str(args.batch_seeds), "--seed", str(seed),
                       "--output", str(directory), "--time-budget", str(remaining - 30),
                       "--drop-offset", *map(str, offset)]
            status["active_batch"] = {"index": index, "split": split,
                                      "seed": seed, "offset_m": offset,
                                      "path": str(directory)}
            checkpoint()
            print(f"Starting batch {index}: {split} seeds {seed}–{seed + args.batch_seeds - 1}, "
                  f"drop={offset}, remaining={remaining / 3600:.2f}h", flush=True)
            started = time.monotonic()
            with (args.output / f"{index:05d}_{split}.log").open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = -signal.SIGTERM
            paired = code == 0 and (directory / "report.json").exists()
            # The benchmark validates every recording before writing each result.
            # Natural early skill failures can prevent pairing an intervention;
            # complete, validated recordings still belong in a failure dataset.
            results_path = directory / "results.json"
            results = json.loads(results_path.read_text()) if results_path.exists() else {}
            valid_collection = (code in (0, 1) and len(results) == 4
                                and all(r["episodes"] == args.batch_seeds for r in results.values()))
            if not valid_collection:
                status["excluded_batches"].append({**status.pop("active_batch"), "returncode": code})
                consecutive_errors += 1
                counts[split] += args.batch_seeds
                checkpoint()
                if consecutive_errors >= 3:
                    raise RuntimeError("Three consecutive invalid batches; inspect batch logs")
                print(f"Excluded batch {index}: exit={code}; advancing to fresh seeds", flush=True)
                index += 1
                continue
            consecutive_errors = 0
            batch_manifest = build_manifest({split: [directory]})
            if len(batch_manifest["episodes"]) != 4 * args.batch_seeds:
                raise RuntimeError("Unexpected batch episode count")
            if {r["seed"] for r in batch_manifest["episodes"]} != seeds:
                raise RuntimeError("Unexpected batch seeds")
            records.extend(batch_manifest["episodes"])
            reserved.update(seeds)
            elapsed = time.monotonic() - started
            durations.append(elapsed)
            status["batches"].append({**status.pop("active_batch"), "wall_time_s": elapsed,
                                      "matched_states_verified": paired})
            counts[split] += args.batch_seeds
            checkpoint()
            print(f"Completed batch {index}: {status['summary']}", flush=True)
            index += 1
        if stop_requested:
            reason = "stop_requested"
        status.update(state="complete", stop_reason=reason)
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        checkpoint()
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
