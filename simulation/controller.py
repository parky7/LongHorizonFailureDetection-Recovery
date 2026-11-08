"""Bounded oracle-driven skill sequencing, independent of the simulator."""
from dataclasses import dataclass

SKILLS = ("approach_object", "grasp_object", "lift_object", "move_to_target", "release_object")


@dataclass
class Outcome:
    success: bool
    recovery_attempts: int
    failures_detected: int


def run_plan(execute, *, after_skill=None, on_event=None, max_retries=0, policy="scripted"):
    """Callbacks operate on one live episode; this function never resets it.

    execute(skill, attempt) must execute and check the skill postcondition.
    after_skill may introduce a disturbance and must return the resulting
    postcondition. False triggers a fresh approach from the actual scene.
    Runtime exceptions propagate: infrastructure errors are not task failures.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    if policy not in ("stop", "continue", "retry", "scripted"):
        raise ValueError(f"Unknown recovery policy: {policy}")
    attempt, failures, index = 0, 0, 0
    while index < len(SKILLS):
        skill = SKILLS[index]
        passed = execute(skill, attempt)
        if passed and after_skill is not None:
            passed = after_skill(skill, attempt)
        if passed:
            index += 1
            continue
        failures += 1
        if on_event is not None:
            on_event("failure_detected", skill, attempt)
        if policy == "continue":
            index += 1
            if index == len(SKILLS):
                return Outcome(False, attempt, failures)
            continue
        if policy == "stop" or attempt >= max_retries:
            return Outcome(False, attempt, failures)
        attempt += 1
        if on_event is not None:
            on_event("recovery_started", skill, attempt)
        if policy == "scripted":
            index = 0
    return Outcome(True, attempt, failures)
