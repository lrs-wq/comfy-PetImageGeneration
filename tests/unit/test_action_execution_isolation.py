from __future__ import annotations

import unittest

from src.comfy_pet_pipeline.api.local_video_parts.action_execution import (
    ActionStageFailure,
    IsolatedActionStageError,
    build_action_execution_scopes,
    failures_by_action,
    isolate_action_stage,
    isolated_action_config,
    isolated_effective_action_config,
    raise_for_action_stage_failures,
)


class CancelledForTest(Exception):
    pass


class ActionExecutionIsolationTests(unittest.TestCase):
    def test_species_and_action_scopes_never_alias(self) -> None:
        cat = build_action_execution_scopes(
            species="cat",
            requested_actions=["eating", "grooming"],
        )
        dog = build_action_execution_scopes(
            species="dog",
            requested_actions=["eating", "standing"],
        )

        self.assertEqual(
            [scope.isolation_key for scope in cat],
            ["cat:eating", "cat:grooming"],
        )
        self.assertEqual(
            [scope.isolation_key for scope in dog],
            ["dog:eating", "dog:standing"],
        )
        self.assertTrue(
            set(scope.isolation_key for scope in cat).isdisjoint(
                scope.isolation_key for scope in dog
            )
        )

    def test_execution_order_does_not_change_request_identity(self) -> None:
        scopes = build_action_execution_scopes(
            species="dog",
            requested_actions=["eating", "standing"],
            execution_actions=["standing", "eating"],
        )

        self.assertEqual(
            [
                (scope.action, scope.request_index, scope.execution_index)
                for scope in scopes
            ],
            [("standing", 2, 1), ("eating", 1, 2)],
        )

    def test_action_config_is_a_deep_snapshot(self) -> None:
        effective = {
            "actions": {
                "eating": {"pose_strength": 1.0, "nested": {"mode": "bowl"}},
                "standing": {
                    "pose_strength": 0.8,
                    "nested": {"mode": "tongue"},
                },
            }
        }

        eating = isolated_action_config(effective, "eating")
        standing = isolated_action_config(effective, "standing")
        eating["pose_strength"] = 9.0
        eating["nested"]["mode"] = "changed"

        self.assertEqual(standing["pose_strength"], 0.8)
        self.assertEqual(standing["nested"]["mode"], "tongue")
        self.assertEqual(effective["actions"]["eating"]["pose_strength"], 1.0)
        self.assertEqual(effective["actions"]["eating"]["nested"]["mode"], "bowl")

        eating_effective = isolated_effective_action_config(effective, "eating")
        self.assertEqual(list(eating_effective["actions"]), ["eating"])
        eating_effective["actions"]["eating"]["nested"]["mode"] = "again"
        self.assertEqual(effective["actions"]["eating"]["nested"]["mode"], "bowl")

    def test_failed_action_does_not_stop_siblings(self) -> None:
        scopes = build_action_execution_scopes(
            species="cat",
            requested_actions=["eating", "grooming"],
        )
        failures: list[ActionStageFailure] = []
        executed: list[str] = []

        for scope in scopes:
            with isolate_action_stage(
                scope=scope,
                stage="scail",
                failures=failures,
            ):
                executed.append(scope.action)
                if scope.action == "eating":
                    raise RuntimeError("isolated eating failure")

        self.assertEqual(executed, ["eating", "grooming"])
        self.assertEqual(list(failures_by_action(failures)), ["eating"])
        with self.assertRaises(IsolatedActionStageError) as raised:
            raise_for_action_stage_failures("scail", failures)
        self.assertEqual(
            raised.exception.failures[0].scope.isolation_key,
            "cat:eating",
        )

    def test_cancellation_still_stops_the_batch(self) -> None:
        scope = build_action_execution_scopes(
            species="dog",
            requested_actions=["eating"],
        )[0]
        failures: list[ActionStageFailure] = []

        with self.assertRaises(CancelledForTest):
            with isolate_action_stage(
                scope=scope,
                stage="qwen_reference",
                failures=failures,
                fatal_exceptions=(CancelledForTest,),
            ):
                raise CancelledForTest("stop")

        self.assertEqual(failures, [])

    def test_plan_rejects_duplicate_or_foreign_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            build_action_execution_scopes(
                species="cat",
                requested_actions=["eating", "eating"],
            )
        with self.assertRaisesRegex(ValueError, "subset"):
            build_action_execution_scopes(
                species="dog",
                requested_actions=["eating", "standing"],
                execution_actions=["grooming"],
            )


if __name__ == "__main__":
    unittest.main()
