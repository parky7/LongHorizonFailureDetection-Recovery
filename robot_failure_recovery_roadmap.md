# Roadmap: Closed-Loop Failure Detection and Recovery for Robotic Manipulation

**Recommended starting point:** ManiSkill 3  
**Suggested duration:** 8–10 weeks  
**Available computing:** NVIDIA A40 and A100 GPUs

## Project objective

Build a closed-loop system that detects failures during robotic manipulation and adapts its plan without resetting the episode. The proposed architecture combines:

1. A **lightweight failure detector** operating on RGB-D observations, robot state, and action/skill history.
2. A **vision-language model (VLM)** that diagnoses an observed failure and proposes corrective high-level steps.
3. A **skill executor** and **ManiSkill simulation environment** to execute plans and collect trajectories.

The key implementation decision is to **avoid training a robot controller, failure detector, and VLM simultaneously**. Start with existing scripted or motion-planned skills, implement a complete recovery loop with oracle labels, then replace the components one at a time.

## 1. First working version (MVP)

Start with a simplified pick-and-place task:

1. Locate and grasp a cube.
2. Lift the cube.
3. Move it toward a target.
4. Release it at the target.
5. Verify that the task succeeded.

Introduce a controlled disturbance—for example, the cube slipping during transport or being displaced before grasping. The system should identify the failed subtask and execute a corrective sequence **from the robot's current state**, without restarting the episode.

### Proposed system architecture

```text
Natural-language goal + high-level plan
                  |
                  v
       Low-level skill executor
   (approach, grasp, lift, move, place)
                  |
                  v
         ManiSkill simulation
    (RGB-D, joint states, actions)
                  |
                  v
           Failure detector
                  |
          +-------+-------+
          |               |
       No failure       Failure
          |               |
          v               v
    Continue plan    VLM diagnosis
                        + replanning
                            |
                            v
                   Validate skill plan
                            |
                            v
                     Skill executor
```

The VLM should produce **high-level skill instructions**, not raw joint commands. A fixed library of executable skills translates its output into motor actions. This keeps execution testable and reduces hallucination-related failures.

## 2. GPU allocation: A40 vs. A100

| GPU | Recommended use |
|---|---|
| **A40 (48 GB)** | ManiSkill simulation, RGB-D rendering, synthetic failure trajectory generation |
| **A100 (40 or 80 GB variants)** | PyTorch training, visual encoder inference, VLM inference or fine-tuning |

**Practical consideration:** ManiSkill visual rendering uses Vulkan. The A40 is a more natural choice for GPU rendering; the A100 is compute-oriented and may not support the required graphics stack. Verify that Vulkan and the drivers are available on the allocated HPC node.

**Preferred setup:** Generate simulation datasets on the A40 and train detectors/run larger multimodal models on the A100. If the GPUs are on different nodes, save trajectories to shared storage and schedule data generation and training as separate jobs.

## 3. Implementation roadmap

| Phase | Timeline | Primary deliverable |
|---|---|---|
| 1. Simulator and skills | Weeks 1–2 | Working ManiSkill task with motion-planned manipulation and logging |
| 2. Scripted recovery | Week 3 | End-to-end failure, detection, replanning, and recovery without reset |
| 3. Failure dataset | Weeks 4–5 | Labeled RGB-D/proprioceptive trajectories with controlled disturbances |
| 4. Learned detector | Weeks 6–7 | Trained and validated failure classification model |
| 5. VLM and longer tasks | Weeks 8–9 | Detector-to-VLM-to-executor pipeline on multiple tasks |
| 6. Evaluation | Week 10 | Benchmark results, ablations, and analysis |

### Phase 1 — Get ManiSkill and a reliable expert policy working

**Weeks 1–2**

Use ManiSkill's existing Panda-arm motion-planning examples rather than training an action policy from scratch. The existing solutions use `mplib` for manipulation tasks such as `PickCube` and `StackCube`.

```bash
pip install -U mani_skill torch

# Generate expert trajectories
python -m mani_skill.examples.motionplanning.panda.run \
    -e "PickCube-v1" -n 10 --save-video

# Download existing demonstrations
python -m mani_skill.utils.download_demo "PickCube-v1"

# Inspect available motion-planning options
python -m mani_skill.examples.motionplanning.panda.run -h
```

*These are starting commands; check the installed ManiSkill version and CLI help because example paths/options may differ between releases.*

Build a wrapper that exposes reusable high-level skills:

- `approach_object()`
- `grasp_object()`
- `lift_object()`
- `move_to_target()`
- `release_object()`

Initially, use privileged simulator state to check whether each skill succeeded. This is an **oracle monitor**, not the final learned detector.

**Deliverable:** A reproducible pick-and-place simulation with saved RGB-D observations, joint states, actions, skill boundaries, and completion labels.

### Phase 2 — Implement a complete scripted recovery loop

**Week 3**

Introduce controlled failures at specific points in the trajectory. For instance, after a successful grasp, displace the cube so the robot loses its grip. The oracle monitor should identify the dropped object and trigger a new approach/grasp sequence from the new location.

Use a finite-state controller to track the current skill, expected postcondition, and recovery state.

```python
while not task_complete and recovery_attempts < MAX_RETRIES:
    skill = plan[current_step]
    execute(skill)

    observation = get_observation()
    failed = oracle_failure_check(skill, observation)

    if failed:
        recovery_attempts += 1
        plan = scripted_recovery(skill, observation)
        current_step = 0
    else:
        current_step += 1
```

The recovery planner must work from the **actual current scene**, not replay the original action sequence. If the object has moved, the original grasp pose may be invalid.

**Deliverable:** A full demonstration of failure injection → detection → replanning → recovery without episode reset.

### Phase 3 — Generate a failure detection dataset

**Weeks 4–5**

Create a purpose-built dataset in ManiSkill to control failure conditions and obtain precise ground-truth labels. Combine successful expert rollouts, imperfect rollouts, and deliberately perturbed scenarios.

| Failure type | Generation method | Expected detector behavior |
|---|---|---|
| Missed grasp | Offset object or grasp pose | Detect that the gripper has not acquired the object |
| Dropped object | Apply a displacement during transport | Detect object/gripper separation |
| Incorrect placement | Perturb placement target | Detect failure to place in the goal region |
| Unexpected scene change | Move an object between skills | Detect an invalidated plan assumption |
| Motion failure | Interrupt or perturb movement | Detect insufficient expected progress |
| Successful execution | Use unperturbed demonstrations | Avoid false alarms |

**Important:** A disturbance is not automatically a task failure. Label the episode based on whether the intended skill postcondition was violated, using simulator ground truth.

**Initial data target:** Approximately **2,000–5,000 trajectories**, including both successes and failures. Record at least:

- RGB-D observations and timestamps
- Robot joint positions, velocities, gripper state
- Actions and active skill
- Ground-truth object poses and task-state variables (for labeling only)
- Failure type and onset time
- Whether recovery was possible/successful

Split by **complete episode**, not frame, to avoid train/test leakage. Hold out object positions, scene configurations, or disturbance intensities to test generalization.

**Deliverable:** A clean, labeled dataset suitable for binary and multiclass failure detection.

### Phase 4 — Train a failure detector

**Weeks 6–7**

Compare three levels of detection complexity:

| Model | Inputs | Purpose |
|---|---|---|
| Rule-based oracle | Privileged simulator state | Upper reference for detection |
| ResNet-18 + MLP | RGB frames, robot states, current skill | Main learned baseline |
| V-JEPA 2 embeddings + MLP | Short trajectory, actions, skill | Research extension |

For ResNet, process a recent frame or a short visual window, then concatenate the visual embedding with proprioception and a learned skill embedding.

A useful starting formulation is:

\[
P(\mathrm{failure}\mid o_{t-k:t},a_{t-k:t},s_t)
\]

Here, \(o\) denotes visual and proprioceptive observations, \(a\) the recent actions, and \(s_t\) the current skill.

Start with **binary classification** (failure/no failure), then optionally add a failure-type classifier.

For V-JEPA, initially **freeze the pretrained encoder and train only a small classification head**. Building a full action-conditioned predictive world model is an additional research effort, not a dependency for the first version.

**Deliverable:** A model that detects failures from observations at inference time, without privileged simulator-state inputs.

### Phase 5 — Integrate VLM reasoning and scale up

**Weeks 8–9**

Once learned failure detection is reliable, invoke a VLM only when the detector reports a likely failure.

Provide the VLM with:

- Natural-language task goal
- Current and previous skills
- Recent RGB observations (and depth-derived summaries)
- Relevant robot-state summaries
- Actions already attempted
- Allowed high-level skills

Candidate models to benchmark include **Qwen3-VL-4B-Instruct** and **Qwen2.5-VL-7B-Instruct**. Start with inference and prompt engineering, not fine-tuning.

Require a structured plan:

```json
{
  "failure_type": "dropped_object",
  "reason": "The cube is no longer held by the gripper.",
  "recovery_plan": [
    "approach_object",
    "grasp_object",
    "lift_object",
    "move_to_target",
    "release_object"
  ]
}
```

Before executing, validate every returned skill against the known skill library and its preconditions. Include retry limits and safeguards against repeated ineffective plans.

After initial success on `PickCube`, progress to `StackCube`, then a longer-horizon environment.

**Deliverable:** A complete learned failure detector → VLM → validated skill executor pipeline, demonstrated on at least two tasks.

### Phase 6 — Evaluation and ablation study

**Week 10**

Compare fixed execution, simple retries, scripted recovery, and learned/VLM-based recovery under matched starting conditions and disturbances.

**Deliverable:** Quantitative benchmark tables, error analysis, ablation results, and example recovery trajectories.

## 4. Recommended datasets and simulations

### 1. ManiSkill 3 built-in tasks — start here

**Tasks:** `PickCube-v1`, `StackCube-v1`, `PegInsertionSide-v1`, `StackPyramid-v1`.

**Why:** Built-in task conditions, expert motion-planning examples, and demonstrations make controlled failure injection and initial debugging comparatively straightforward.

**Priority:** Begin with `PickCube`; use `StackCube` for more meaningful grasp-and-placement verification.

### 2. ManiSkill-HAB — best long-horizon extension

**Tasks:** `TidyHouse`, `PrepareGroceries`, `SetTable`.

**Why:** Multi-stage household manipulation with skills such as pick, place, open, and close. Earlier subtasks alter the conditions faced by later subtasks, making it relevant to failure recovery over long horizons.

**Caution:** Task sequences and datasets can be large; start with a curated subset rather than downloading everything.

### 3. LIBERO-Long — external benchmark

**Why:** Language-conditioned long-horizon manipulation tasks with visual observations, proprioception, and demonstrations. It is a useful external testbed for detection and recovery generalization.

**Caution:** It uses a different simulator stack; don't integrate it until the main ManiSkill system is working.

### 4. RoboCasa365 — richer language-conditioned tasks

**Why:** Large-scale kitchen manipulation with atomic and composite tasks, plus human and synthetic demonstrations. Useful for later studies of failure recovery in complex, language-conditioned multi-step tasks.

**Caution:** Too broad for the initial prototype.

### 5. CALVIN — additional benchmark option

**Why:** Language-conditioned sequences of manipulation tasks, useful for evaluating multi-step completion and recovery.

**Caution:** Better as a later experiment than a starting environment.

### Recommended progression

**PickCube → StackCube → ManiSkill-HAB**, with LIBERO-Long as an optional cross-environment evaluation.

## 5. Experimental design

The central research question should be:

> Does intelligent failure detection and replanning improve task success over longer horizons, compared with fixed execution and simpler recovery baselines?

### Ablation configurations

| System | Failure detection | Recovery mechanism |
|---|---|---|
| A | None | Continue original policy |
| B | Execution status / basic monitor | Simple skill retry |
| C | Oracle (simulator state) | Scripted recovery |
| D | Learned ResNet | Scripted recovery |
| E | Learned ResNet | VLM-generated recovery |
| F | V-JEPA visual features | VLM-generated recovery |

C provides a reference for the effect of near-perfect monitoring. D vs. E helps isolate the added value of VLM-based replanning.

### Metrics

1. **Task success rate:** Percentage of episodes reaching the final goal.
2. **Failure detection F1:** Precision and recall on failure events.
3. **False alarms per episode:** Important as the horizon length increases.
4. **Detection latency:** Delay from the onset of an actual failure to its detection.
5. **Recovery success rate:** Fraction of recoverable failures resolved without resetting.
6. **Recovery cost:** Additional actions, wall-clock time, and VLM inference time.

As an initial evaluation target, run at least **100 held-out episodes per condition**, under matched scenarios and random seeds. Increase repetitions if differences between systems are small or variability is high.

## 6. Highest-value research extension: reactive vs. predictive detection

A standard visual detector may identify that a cube **has already fallen**. A predictive model might detect divergence from expected execution *before an obvious final failure occurs*.

Compare:

**Reactive detection**

\[
P(f_t\mid o_{0:t},a_{0:t})
\]

**Predictive detection**

\[
P(f_{t:t+H}\mid o_{0:t},a_{0:t+H})
\]

Here, \(H\) is a short prediction horizon, and the future action sequence represents **planned** actions.

The Foresight work is relevant because it explores action-conditioned visual prediction and disagreement between expected and observed dynamics.

Keep this as an **optional research extension**, not a requirement for the MVP.

Other useful literature directions:

- **AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation** — natural-language diagnosis of manipulation failures.
- **Inner Monologue: Embodied Reasoning through Planning with Language Models** — feedback-informed replanning.
- **Foresight: Failure Detection for Long-Horizon Robotic Manipulation** — predictive failure monitoring.
- **FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement** — recovery and retry baselines.
- **ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills** — simulation and demonstration background.

## 7. Suggested repository organization

```text
robot-failure-recovery/
├── configs/
├── simulation/
│   ├── env_wrapper.py
│   ├── skills.py
│   └── perturbations.py
├── data/
│   ├── collect.py
│   ├── label.py
│   └── dataset.py
├── detectors/
│   ├── oracle.py
│   ├── resnet.py
│   └── temporal_encoder.py
├── planning/
│   ├── vlm_planner.py
│   ├── prompt_templates.py
│   └── plan_validator.py
├── controller/
│   └── closed_loop.py
├── evaluation/
│   ├── run_benchmark.py
│   └── metrics.py
└── scripts/
    ├── train_detector.py
    └── evaluate.py
```

Plan HPC jobs separately for trajectory generation, detector training, and VLM evaluation. Keep configurations, random seeds, pretrained model versions, and dataset split definitions under version control.

## 8. First-week checklist

- [x] Install ManiSkill 3 on a GPU node and verify RGB-D rendering.
- [x] Run a provided `PickCube` motion-planning example successfully.
- [x] Save at least ten successful trajectories with observations and actions.
- [x] Identify which Panda motion-planner functions can be exposed as reusable skills.
- [x] Inject a dropped-object disturbance and confirm the simulator reports the actual resulting task state.

Validated November 6, 2026 in `learningProject` on an RTX PRO 6000 Blackwell
MIG allocation. `outputs/preflight.json` records the upstream expert and native
RGB-D check; `outputs/phase1_v1/validation.json` records ten successful tabletop
placements (seeds 100–109). The custom tabletop variant is not the standard
PickCube benchmark. `outputs/recovery_smoke_v1` demonstrates one detected drop
and successful scripted recovery without resetting. See the README for exact
commands, dependency pins, recording format and comparison policies.

The completed [validation report](reports/phase1_validation_2026-11-06.md)
also records 100 matched test seeds per condition: nominal and oracle scripted
recovery each succeeded 100/100, while fixed-order execution and simple retries
after a drop each succeeded 0/100. All 400 recordings and paired states were
validated. Phase 2's oracle recovery milestone is demonstrated for this one
controlled-drop scenario. A separate joint-aperture baseline is evaluated
offline; replacing the oracle online, broadening failures and increasing the
task horizon remain open.

**Week 3 milestone:** A controlled failure should be detected by an oracle monitor, followed by an executable recovery plan that successfully completes the task without restarting the episode.

## Final recommendation

Invest most of the initial effort in **reliable failure injection, ground-truth labeling, and reusable recovery skills**. Once those pieces are working, you can replace the oracle monitor with a learned detector and add VLM planning without rebuilding the rest of the system.

---

*Roadmap adapted from the project proposal “Learning for Interactive Robots” (October 2026) and the subsequent implementation plan. Library/model availability and example CLI commands should be verified against the versions installed on the HPC cluster.*
