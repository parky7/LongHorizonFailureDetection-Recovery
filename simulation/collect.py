"""Run with python -m simulation.collect; records every control step to HDF5."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--time-budget", type=float, default=1800,
                        help="Stop starting new episodes after this many wall-clock seconds")
    parser.add_argument("--disturbance", choices=("none", "drop_after_lift"), default="none")
    parser.add_argument("--drop-offset", type=float, nargs=2, default=(0.08, 0.04),
                        metavar=("X", "Y"), help="Controlled drop displacement in meters")
    parser.add_argument("--recovery", choices=("stop", "continue", "retry", "scripted"), default="stop")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.time_budget <= 0:
        parser.error("--time-budget must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    import math
    if not all(math.isfinite(v) and abs(v) <= 0.2 for v in args.drop_offset):
        parser.error("--drop-offset components must be finite and within +/-0.2 meters")
    # Keep --help usable even before simulator dependencies are installed.
    import gymnasium as gym
    import h5py
    import numpy as np
    import sapien
    import mani_skill.envs
    from simulation.skills import PandaSkills, MotionPlanningFailure
    from simulation.oracle import check
    from simulation.controller import run_plan
    from simulation.perturbations import drop_object

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
            if time.monotonic() - started >= args.time_budget:
                raise TimeoutError("Collection time budget reached during episode")
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
    for package in ("mani_skill", "mani_skill-nightly", "sapien", "mplib", "torch", "numpy", "h5py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    env = Recorder(gym.make("PickCube-v1", num_envs=1, obs_mode="rgbd",
                           control_mode="pd_joint_pos", sim_backend="cpu",
                           max_episode_steps=1000))
    results = []
    started = time.monotonic()
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            if time.monotonic() - started >= args.time_budget:
                print("Collection time budget reached", flush=True)
                break
            # Exclusive creation protects earlier runs from accidental overwrite.
            with h5py.File(args.output / f"episode_{seed:06d}.h5", "x") as file:
                episode_started = time.monotonic()
                env.file, env.steps, env.skill = file, 0, "reset"
                # Rebuild between episodes to clear PhysX contact caches. Otherwise
                # an earlier recovery can change the next episode's pre-drop state.
                env.reset(seed=seed, options={"reconfigure": True})
                base = env.unwrapped
                # Explicit variant: a reachable tabletop goal, not the floating benchmark goal.
                cube = base.cube.pose.sp.p
                target = [float(cube[0]), float(cube[1] + (0.15 if cube[1] <= 0 else -0.15)), float(base.cube_half_size)]
                base.goal_site.set_pose(sapien.Pose(target))
                file.attrs.update(seed=seed, task="PickCube-tabletop-placement-v0", status="running",
                                  schema_version=2, disturbance=args.disturbance, recovery=args.recovery,
                                  max_retries=args.max_retries, reset_count=1,
                                  drop_offset_m=args.drop_offset,
                                  reset_reconfigured=True, depth_units="millimeters",
                                  goal_thresh=base.goal_thresh,
                                  versions=json.dumps(versions), control_freq=base.control_freq)
                initial = snapshot(base)
                for key, value in initial.items():
                    file.create_dataset("initial/oracle/" + key, data=value)
                skills = None
                success = False
                outcome = None
                try:
                    skills = PandaSkills(env)
                    injected = False
                    boundary_count = 0

                    def record_execution(name, attempt, operation, postcondition, role="skill"):
                        nonlocal boundary_count
                        env.skill = name
                        start = env.steps
                        boundary = file.create_group(f"skills/{boundary_count:04d}")
                        boundary_count += 1
                        boundary.attrs.update(name=name, attempt=attempt, role=role)
                        passed = False
                        try:
                            operation()
                            passed = bool(postcondition())
                        except MotionPlanningFailure as error:
                            boundary.attrs["planning_error"] = str(error)
                        finally:
                            boundary.attrs.update(start_step=start, end_step_exclusive=env.steps, success=passed)
                            file.flush()
                        print(f"seed={seed} attempt={attempt} skill={name} role={role} passed={passed} steps={env.steps}", flush=True)
                        return passed

                    def postcondition(name):
                        return check(name, snapshot(base), initial["cube_pose"][2], base.goal_thresh)

                    def execute(name, attempt):
                        return record_execution(name, attempt, getattr(skills, name), lambda: postcondition(name))

                    def event(kind, name, attempt):
                        group = file.create_group(f"events/{len(file.get('events', {})):04d}")
                        group.attrs.update(kind=kind, skill=name, attempt=attempt, step=env.steps)
                        return group

                    def after_skill(name, attempt):
                        nonlocal injected
                        if args.disturbance != "drop_after_lift" or injected or name != "lift_object":
                            return True
                        injected = True
                        group = event("disturbance", name, attempt)
                        group.attrs.update(type="drop_after_lift", onset_step=env.steps,
                                           offset_m=args.drop_offset)
                        for key, value in snapshot(base).items():
                            group.create_dataset("before/" + key, data=value)

                        def displaced():
                            for key, value in snapshot(base).items():
                                group.create_dataset("after_intervention/" + key, data=value)

                        # Keep intervention identity out of the active-skill input
                        # supplied to a future detector. It belongs in oracle events.
                        passed = record_execution(name, attempt,
                                                  lambda: drop_object(env, skills.planner, displaced,
                                                                      offset=args.drop_offset),
                                                  lambda: postcondition(name), role="intervention")
                        group.attrs.update(detection_step=env.steps, postcondition_passed=passed)
                        for key, value in snapshot(base).items():
                            group.create_dataset("after_settle/" + key, data=value)
                        return passed

                    outcome = run_plan(execute, after_skill=after_skill, on_event=event,
                                       max_retries=args.max_retries, policy=args.recovery)
                    success = outcome.success
                except Exception as error:
                    file.attrs["error"] = f"{type(error).__name__}: {error}"
                    print(f"seed={seed}: {error}")
                finally:
                    if skills is not None:
                        skills.planner.close()
                file.attrs["success"] = success
                file.attrs["status"] = "complete"
                file.attrs["recovery_attempts"] = outcome.recovery_attempts if outcome else 0
                file.attrs["failures_detected"] = outcome.failures_detected if outcome else 0
                file.attrs["wall_time_s"] = time.monotonic() - episode_started
                results.append({"seed": seed, "success": success, "steps": env.steps,
                                "recovery_attempts": int(file.attrs["recovery_attempts"]),
                                "failures_detected": int(file.attrs["failures_detected"]),
                                "wall_time_s": float(file.attrs["wall_time_s"])})
                (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
                print(results[-1], flush=True)
    finally:
        env.close()
    print(f"Successful episodes: {sum(r['success'] for r in results)}/{len(results)}")
    if len(results) != args.episodes or not all(r["success"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
