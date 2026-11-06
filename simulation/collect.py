"""Run with python -m simulation.collect; records every control step to HDF5."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    # Keep --help usable even before simulator dependencies are installed.
    import gymnasium as gym
    import h5py
    import numpy as np
    import sapien
    import mani_skill.envs
    from simulation.skills import PandaSkills
    from simulation.oracle import check

    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def snapshot(base):
        info = base.evaluate()
        return {"qpos": array(base.agent.robot.get_qpos())[0],
                "qvel": array(base.agent.robot.get_qvel())[0],
                "cube_pose": array(base.cube.pose.raw_pose)[0],
                "goal_pose": array(base.goal_site.pose.raw_pose)[0],
                "tcp_pose": array(base.agent.tcp.pose.raw_pose)[0],
                "is_grasped": bool(array(info["is_grasped"]).item()),
                "success": bool(array(info["success"]).item())}

    class Recorder(gym.Wrapper):
        def step(self, action):
            if self.steps >= 1000:
                raise RuntimeError("Episode exceeded 1000 control steps")
            obs, reward, terminated, truncated, info = super().step(action)
            group = self.file.create_group(f"steps/{self.steps:06d}")
            group.attrs["skill"] = self.skill
            group.attrs["simulation_time_s"] = (self.steps + 1) / self.unwrapped.control_freq
            group.create_dataset("action", data=array(action))
            for key, value in snapshot(self.unwrapped).items():
                group.create_dataset("oracle/" + key, data=value)
            group.create_dataset("reward", data=array(reward))
            group.attrs["terminated"] = bool(array(terminated).item())
            group.attrs["truncated"] = bool(array(truncated).item())
            for camera, images in obs["sensor_data"].items():
                for modality in ("rgb", "depth"):
                    group.create_dataset(f"observations/{camera}/{modality}",
                                         data=array(images[modality])[0], compression="gzip")
            self.steps += 1
            if bool(array(truncated).item()):
                raise RuntimeError("Episode truncated")
            return obs, reward, terminated, truncated, info

    args.output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("mani_skill", "mani_skill-nightly", "sapien", "mplib", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    env = Recorder(gym.make("PickCube-v1", num_envs=1, obs_mode="rgbd",
                           control_mode="pd_joint_pos", sim_backend="cpu",
                           max_episode_steps=1000))
    results = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            # Exclusive creation protects earlier runs from accidental overwrite.
            with h5py.File(args.output / f"episode_{seed:06d}.h5", "x") as file:
                env.file, env.steps, env.skill = file, 0, "reset"
                env.reset(seed=seed)
                base = env.unwrapped
                # Explicit variant: a reachable tabletop goal, not the floating benchmark goal.
                cube = base.cube.pose.sp.p
                target = [float(cube[0]), float(cube[1] + (0.15 if cube[1] <= 0 else -0.15)), float(base.cube_half_size)]
                base.goal_site.set_pose(sapien.Pose(target))
                file.attrs.update(seed=seed, task="PickCube-tabletop-placement-v0",
                                  versions=json.dumps(versions), control_freq=base.control_freq)
                initial = snapshot(base)
                for key, value in initial.items():
                    file.create_dataset("initial/oracle/" + key, data=value)
                skills = PandaSkills(env)
                success = False
                try:
                    for name in ("approach_object", "grasp_object", "lift_object", "move_to_target", "release_object"):
                        env.skill = name
                        start = env.steps
                        getattr(skills, name)()
                        passed = check(name, snapshot(base), initial["cube_pose"][2], base.goal_thresh)
                        boundary = file.create_group("skills/" + name)
                        boundary.attrs.update(start_step=start, end_step_exclusive=env.steps, success=passed)
                        if not passed:
                            break
                    else:
                        success = True
                except Exception as error:
                    file.attrs["error"] = f"{type(error).__name__}: {error}"
                    print(f"seed={seed}: {error}")
                finally:
                    skills.planner.close()
                file.attrs["success"] = success
                results.append({"seed": seed, "success": success, "steps": env.steps})
                print(results[-1])
    finally:
        env.close()
    print(f"Successful episodes: {sum(r['success'] for r in results)}/{len(results)}")
    if not all(r["success"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
