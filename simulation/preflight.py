"""Bounded dependency, RGB-D and upstream expert check before data collection."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/preflight.json"))
    args = parser.parse_args()
    import gymnasium as gym
    import mani_skill.envs
    import torch
    from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve

    started = time.monotonic()
    report = {"versions": {p: importlib.metadata.version(p) for p in
                           ("mani_skill", "sapien", "mplib", "torch", "numpy", "h5py")},
              "cuda_available": torch.cuda.is_available()}
    env = None
    def checkpoint():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        print("Creating CPU-physics, RGB-D environment", flush=True)
        env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgbd",
                       control_mode="pd_joint_pos", sim_backend="cpu", max_episode_steps=1000)
        obs, _ = env.reset(seed=0)
        report["cameras"] = {}
        for camera, images in obs["sensor_data"].items():
            rgb, depth = images["rgb"], images["depth"]
            assert rgb.shape[-1] == 3 and depth.shape[-1] == 1
            assert torch.isfinite(depth).all() and (depth > 0).any()
            report["cameras"][camera] = {
                "rgb_shape": list(rgb.shape), "rgb_dtype": str(rgb.dtype),
                "depth_shape": list(depth.shape), "depth_dtype": str(depth.dtype),
                "depth_max": float(depth.max())}
        assert report["cameras"], "No RGB-D cameras returned"
        checkpoint()
        print("RGB-D check passed; running upstream expert", flush=True)
        result = solve(env, seed=0, debug=False, vis=False)
        report["upstream_success"] = not isinstance(result, int) and bool(result[-1]["success"].item())
        assert report["upstream_success"], "Upstream PickCube expert did not succeed"
        report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        report["wall_time_s"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
