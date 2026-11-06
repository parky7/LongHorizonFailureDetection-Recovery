# Robot Failure Detection and Recovery

Phase 1 implements a single-environment Panda motion-planning baseline, reusable
skills, privileged postcondition checks, and RGB-D trajectory collection.
Simulator execution is still pending validation on a configured machine.

## Setup

Use Python 3.11 in a fresh environment. The preferred simulation machine is the
Linux A40 node with working NVIDIA drivers and Vulkan rendering.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build appropriate to the node, then:
python -m pip install torch
python -m pip install -r requirements-sim.txt
python -m mani_skill.examples.motionplanning.panda.run -h
python -m mani_skill.examples.motionplanning.panda.run -e PickCube-v1 -n 1 --save-video
```

On macOS, the project's existing `.venv-maniskill` uses Python 3.9 and currently
lacks SAPIEN and mplib. Create a Python 3.11 environment instead. Follow
[ManiSkill's macOS setup](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/getting_started/macos_install.md)
for `mani_skill-nightly`, SAPIEN, and the Vulkan SDK; install `mplib` separately.
The nightly package alone may not install SAPIEN. macOS supports CPU simulation;
use Linux for the primary data-generation workflow. Motion-planner compatibility
on macOS must also be verified.

## Collect Phase 1 trajectories

From the repository root:

```bash
python -m simulation.collect --episodes 1 --seed 0 --output outputs/smoke
python -m simulation.collect --episodes 10 --seed 100 --output outputs/phase1
```

The collector uses CPU physics and RGB-D rendering. It uses the installed
ManiSkill Panda screw-motion planner, following the upstream PickCube expert.
It deliberately changes the goal to a tabletop location 15 cm from the cube,
so release is meaningful. This variant is recorded as
`PickCube-tabletop-placement-v0`; its results are not the standard PickCube benchmark.
Skills are approach, grasp, lift, move to target (including lowering), and release
(including retraction and settling). No recovery or disturbance is implemented yet.

Every seed gets an HDF5 file; existing files are never overwritten. Each contains:

- `initial/oracle`: initial joint states, cube/goal/TCP poses and oracle labels.
- `steps/NNNNNN/action`, reward, termination/truncation flags, active skill and
  post-action simulation time in seconds.
- `steps/NNNNNN/observations/<camera>/{rgb,depth}`: post-action RGB-D, preserving
  ManiSkill's native dtypes and depth units (millimeters).
- `steps/NNNNNN/oracle`: joint positions/velocities, poses, grasp and task labels.
  Panda's last two joint positions are its finger joints.
- `skills/<name>`: zero-based start and exclusive end indices, and completion label.
- File attributes: seed, task variant, installed package versions and episode success.

Observations are post-action; step zero's action starts from `initial/oracle`.
Privileged state is separated from camera inputs. Failed plans/postconditions are
retained and the process exits nonzero if any episode fails. Ten attempts do not
guarantee ten successes: inspect results and collect more seeds if needed. Runtime
exceptions are saved in the file's `error` attribute. Skill thresholds and the
placement geometry still require simulator calibration before bulk collection.

## Lightweight checks

```bash
python -m unittest discover -s tests -v
python -m simulation.collect --help
```

Next validation gates: verify RGB-D rendering, run the upstream expert, run one
placement episode, inspect the HDF5 data, then collect ten successful episodes.
The roadmap checklist remains open until those simulation checks actually pass.
