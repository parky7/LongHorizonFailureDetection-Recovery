"""Build explicit dataset splits, keeping every condition of a seed together."""
import argparse
import json
from pathlib import Path

import h5py


def build_manifest(splits):
    owners = {}
    records = []
    summary = {}
    for split, roots in splits.items():
        paths = set()
        for root in roots:
            root = Path(root)
            if not root.exists():
                raise ValueError(f"Missing dataset path: {root}")
            paths.update([root.resolve()] if root.is_file()
                         else (p.resolve() for p in root.rglob("episode_*.h5")))
        if not paths:
            raise ValueError(f"No episode files in {split}")
        split_records = []
        for path in sorted(paths):
            with h5py.File(path) as file:
                if file.attrs.get("status") != "complete" or "error" in file.attrs:
                    raise ValueError(f"Incomplete or errored episode: {path}")
                if "success" not in file.attrs or not len(file.get("steps", {})):
                    raise ValueError(f"Missing outcome or control steps: {path}")
                seed = int(file.attrs["seed"])
                # Conservatively group by seed across tasks as well as conditions.
                # This prevents nominal and perturbed versions of one scene from
                # appearing on opposite sides of a train/test split.
                if seed in owners and owners[seed] != split:
                    raise ValueError(f"Seed {seed} leaks across {owners[seed]} and {split}")
                owners[seed] = split
                record = {
                    "path": str(path), "split": split, "seed": seed,
                    "task": str(file.attrs["task"]), "success": bool(file.attrs["success"]),
                    "steps": len(file["steps"]), "schema_version": int(file.attrs.get("schema_version", 1)),
                    "disturbance": str(file.attrs.get("disturbance", "none")),
                    "recovery": str(file.attrs.get("recovery", "stop")),
                }
                split_records.append(record)
        records.extend(split_records)
        summary[split] = {
            "episodes": len(split_records), "unique_seeds": len({r["seed"] for r in split_records}),
            "successes": sum(r["success"] for r in split_records),
            "failures": sum(not r["success"] for r in split_records)}
    return {"grouping": "complete seed across all conditions", "summary": summary, "episodes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument("--" + split, nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest({name: getattr(args, name) for name in ("train", "validation", "test")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Protect an existing split definition from accidental replacement.
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
