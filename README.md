# Robot Failure Detection and Recovery

Phase 1 is validated on a Linux RTX PRO 6000 Blackwell MIG allocation: RGB-D
rendering, the upstream PickCube expert, and ten successful recorded placements
(seeds 100–109). Phase 2 adds controlled drops, oracle detection, and bounded
scripted recovery from the current scene without resetting the episode.
This is a five-skill manipulation prototype; learned detection and longer tasks
remain future phases. See [the roadmap](robot_failure_recovery_roadmap.md).

The [November 6 validation report](reports/phase1_validation_2026-11-06.md)
records 100/100 nominal successes, 0/100 for fixed execution or simple retries
after a drop, and 100/100 with oracle scripted recovery on matched test seeds.
It includes artifact links, costs, the aperture baseline and experiment limits.

## Setup

Use the existing Python 3.11 `learningProject` environment on a Linux node with
working NVIDIA drivers and Vulkan rendering. CPU physics still needs GPU Vulkan
rendering for camera observations. The tested environment is captured in
`requirements-learningProject.lock.txt` (including PyTorch 2.15.0/CUDA 13.2).

```bash
conda activate learningProject
# For a new environment, install a PyTorch build appropriate to the node, then:
python -m pip install torch
python -m pip install -r requirements-sim.txt
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
vulkaninfo --summary
python -u -m simulation.preflight
```

Keep the NumPy/OpenCV pins: `mplib==0.1.1` crashed during planner construction
with NumPy 2.4.6; the same preflight passed with NumPy 1.26.4. Preflight saves
versions, camera shapes/dtypes and the upstream expert result to `outputs/preflight.json`.
GPU slices must expose Vulkan; verify the allocated device rather than assuming
all MIG profiles support graphics. In this workspace the simulator requires
access outside the command sandbox to reach the GPU.

## Collect Phase 1 trajectories

From the repository root:

```bash
python -m simulation.collect --episodes 1 --seed 0 --output outputs/smoke
python -m simulation.collect --episodes 10 --seed 100 --output outputs/phase1
python -m simulation.inspect_data outputs/phase1 --require-successes 10 --report outputs/phase1/validation.json
```

The collector uses CPU physics and RGB-D rendering. It uses the installed
ManiSkill Panda screw-motion planner, following the upstream PickCube expert.
It deliberately changes the goal to a tabletop location 15 cm from the cube,
so release is meaningful. This variant is recorded as
`PickCube-tabletop-placement-v0`; its results are not the standard PickCube benchmark.
Skills are approach, grasp, lift, move to target (including lowering), and release
(including retraction and settling). The physics scene is rebuilt between
episodes to clear contact caches; recovery stays within the existing episode.

Every seed gets an HDF5 file; existing files are never overwritten. Each contains:

- `initial/oracle`: initial joint states, cube/goal/TCP poses and oracle labels.
- `steps/NNNNNN/action`, reward, termination/truncation flags, active skill and
  post-action simulation time in seconds.
- `steps/NNNNNN/observations/<camera>/{rgb,depth}`: post-action RGB-D, preserving
  ManiSkill's native dtypes and depth units (millimeters).
- `steps/NNNNNN/oracle`: joint positions/velocities, poses, grasp and task labels.
  Panda's last two joint positions are its finger joints.
- `skills/NNNN`: skill name, attempt number, zero-based start and exclusive end
  indices, role (`skill` or `intervention`), and postcondition label. Schema 2 permits repeated skills; early
  Phase 1 files used `skills/<name>` and remain readable by the validator.
- `events/NNNN`: intervention, failure detection, and recovery events. Interventions
  include onset/detection step indices and oracle snapshots before displacement,
  immediately after displacement, and after settling.
- File attributes: seed, schema, variant, versions, depth units, goal threshold,
  recovery settings, wall time, completion status and episode success.

Observations are post-action; step zero's action starts from `initial/oracle`.
An explicit intervention changes state between actions; its event snapshots
record that discontinuity. Native images are 128×128 RGB `uint8` and depth `int16`
in millimeters. The goal marker is hidden by ManiSkill's sensor rendering.
Intervention frames retain the actual active skill (`lift_object`); intervention
identity is recorded separately in oracle metadata to avoid a skill-label shortcut.
Privileged state is separated from camera inputs. Failed plans/postconditions are
retained and the process exits nonzero if any episode fails. Ten attempts do not
guarantee ten successes: inspect results and collect more seeds if needed. Runtime
exceptions are saved in the file's `error` attribute; path-planning failures
are skill failures eligible for recovery. `summary.json` updates after every
episode. `--time-budget` (seconds, default 1800) also checks the deadline before
each control step. Each episode has a 1000-step ceiling. Incomplete or corrupt
recordings cannot count as validated successes.

## Scripted recovery and comparisons

```bash
python -u -m simulation.collect --episodes 1 --seed 0 --output outputs/recovery \
  --disturbance drop_after_lift --recovery scripted --max-retries 2
python -m simulation.inspect_data outputs/recovery --require-successes 1
python -m simulation.preview outputs/recovery/episode_000000.h5 --output outputs/recovery/preview
python -u -m simulation.benchmark --episodes 100 --seed 1000 \
  --output outputs/benchmark --time-budget 1800
```

The drop is a controlled 8 cm/4 cm sideways pose displacement after a verified
lift, followed by 16 control steps of settling. It is an intervention rather
than a realistic slip model. Labels come from the observed postcondition.
The oracle checks skill boundaries; it is not a continuous learned detector.
Recovery opens the gripper, recomputes the grasp from the actual cube pose,
and executes the remaining placement sequence with at most two recovery attempts.

`--recovery stop` stops on a failed postcondition (default), `continue` follows
the original skill order, `retry` repeats only the failed skill, and `scripted`
restarts at approach. The benchmark compares nominal execution against the
three latter policies under matched disturbances. It runs sequentially, saves
per-condition logs and HDF5 files, validates recordings, and verifies paired
initial and pre-intervention states. Use a new output directory for each run.
`config.json` and `source/` preserve the run's code and configuration;
`results.json` contains completed conditions and `report.json` marks a completed
paired comparison. Natural task failures are retained in the denominator.

## Dataset split checks

For allocation-bounded collection, `simulation.campaign` interleaves four training
batches with one validation batch, using disjoint seeds starting at 20000 and
2000000. Each batch runs the four benchmark policies on ten scenes and cycles
through six controlled drop offsets (5–12 cm in the larger component). Natural
task failures remain in the dataset. Only complete, structurally validated
batches enter the checkpointed `dataset_manifest.json`; batch status records
whether paired intervention states were also verified. Three consecutive invalid
batches stop the campaign. The process checks free storage before each batch.

```bash
python -u -m simulation.campaign \
  --output /scratch/$USER/longhorizon_campaign_new \
  --test-manifest outputs/dataset_manifest.json \
  --end-time "$((SLURM_JOB_END_TIME - 180))" --batch-seeds 10
```

Use a fresh directory and the existing `learningProject` environment. The
November 6 active run uses a detached process and a frozen source snapshot;
its paths and monitoring commands are in
[the run record](reports/gpu_run_2026-11-06.md). This campaign varies controlled
drops only; missed grasps, placement errors and motion interruptions still need
separate collection methods.

`data.manifest` builds explicit train/validation/test manifests from completed
recordings. It rejects any seed shared across splits, including nominal and
disturbed versions of the same scene. It also rejects interrupted or errored
recordings, while retaining valid task failures. Run the RGB-D validator before
building a manifest; the manifest checks metadata rather than every frame.

```bash
python -m data.manifest --train outputs/phase1_v1 \
  --validation outputs/benchmark_validation_v1 --test outputs/benchmark_100_v2 \
  --output outputs/dataset_manifest.json
```

This example reserves benchmark seeds 1000–1099 for testing. The current Phase 1
training examples contain only successes and are insufficient to train a failure
detector. Collect varied training failures on separate seeds before training.

## Lightweight grasp-loss baseline

`detectors.aperture.ApertureMonitor` uses only Panda joint readings and the
holding skill. It reports grasp loss when the fingers close below 20 mm, half
the known cube width. It returns no judgment for approach/release and cannot
verify lift height, placement accuracy or grasp contact forces. Evaluate it
after skills complete, rather than while an initial grasp is still closing.

```bash
python -m evaluation.aperture outputs/benchmark_100_v2 \
  --output outputs/benchmark_100_v2/aperture_report.json
```

This offline baseline is scored against oracle grasp labels at holding-skill
boundaries and after settled drops. It never receives cube pose, oracle grasp
flags, disturbance identity or episode outcomes as inference inputs. Its
precision/recall describe boundary samples, including repeated observations of
one failure, rather than independent failure events. The closed-loop benchmark
still uses the full oracle monitor; aperture-only online recovery remains a
separate experiment.

## Lightweight checks

```bash
python -m unittest discover -s tests -v
python -m simulation.collect --help
```

The validated Phase 1 data are in `outputs/phase1_v1`; the first recovered
trajectory and RGB-D preview are in `outputs/recovery_smoke_v1`. Outputs are
local artifacts excluded from Git. Next work is to vary failure types and
intensities, establish episode-level dataset splits, and train a small detector
before adding VLM planning or longer-horizon environments.
