import unittest

from simulation.controller import SKILLS, run_plan


class ControllerTests(unittest.TestCase):
    def test_nominal_plan_has_no_recovery(self):
        calls = []
        result = run_plan(lambda skill, attempt: calls.append((skill, attempt)) or True)
        self.assertTrue(result.success)
        self.assertEqual(calls, [(s, 0) for s in SKILLS])
        self.assertEqual(result.recovery_attempts, 0)

    def test_disturbance_restarts_from_approach_in_same_scene(self):
        scene = {"cube": "original", "grasp_sources": []}
        events = []

        def execute(skill, attempt):
            if skill == "approach_object":
                scene["grasp_sources"].append(scene["cube"])
            return True

        def after(skill, attempt):
            if skill == "lift_object" and attempt == 0:
                scene["cube"] = "displaced"
                return False
            return True

        result = run_plan(execute, after_skill=after,
                          on_event=lambda *event: events.append(event), max_retries=1)
        self.assertTrue(result.success)
        self.assertEqual(scene["grasp_sources"], ["original", "displaced"])
        self.assertEqual(result.failures_detected, 1)
        self.assertEqual(events, [("failure_detected", "lift_object", 0),
                                  ("recovery_started", "lift_object", 1)])

    def test_persistent_failure_stops_at_retry_limit(self):
        calls = []
        result = run_plan(lambda *args: calls.append(args) or False, max_retries=2)
        self.assertFalse(result.success)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.recovery_attempts, 2)

    def test_injection_does_not_force_failure_label(self):
        result = run_plan(lambda *_: True, after_skill=lambda *_: True, max_retries=1)
        self.assertTrue(result.success)
        self.assertEqual(result.failures_detected, 0)

    def test_retry_repeats_failed_skill_without_replanning(self):
        calls = []
        def execute(skill, attempt):
            calls.append(skill)
            return skill != "lift_object"
        result = run_plan(execute, max_retries=2, policy="retry")
        self.assertFalse(result.success)
        self.assertEqual(calls, ["approach_object", "grasp_object", "lift_object", "lift_object", "lift_object"])

    def test_fixed_execution_continues_after_failure(self):
        calls = []
        def execute(skill, attempt):
            calls.append(skill)
            return skill not in ("lift_object", "release_object")
        result = run_plan(execute, policy="continue")
        self.assertFalse(result.success)
        self.assertEqual(calls, list(SKILLS))
        self.assertEqual(result.failures_detected, 2)

    def test_execution_error_propagates(self):
        def fail(*_):
            raise RuntimeError("renderer unavailable")
        with self.assertRaisesRegex(RuntimeError, "renderer unavailable"):
            run_plan(fail, max_retries=2)


if __name__ == "__main__":
    unittest.main()
