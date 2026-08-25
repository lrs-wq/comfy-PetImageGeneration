import json
from pathlib import Path
import unittest


CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "profiles"
    / "morphology"
    / "dog_white_model_catalog_v1.json"
)


class DogWhiteModelCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_categories_are_registered_without_changing_live_routing(self) -> None:
        self.assertFalse(self.catalog["routing"]["enabled"])
        self.assertTrue(self.catalog["routing"]["test_platform_selection_enabled"])
        self.assertEqual(
            self.catalog["routing"]["current_fallback_category"],
            "neutral_dog",
        )
        self.assertEqual(
            self.catalog["routing"]["test_platform_profiles"],
            {
                "medium_dog": "dog_medium_white_model_test_v1",
                "small_dog": "dog_small_white_model_test_v1",
            },
        )
        self.assertEqual(
            set(self.catalog["categories"]),
            {"neutral_dog", "small_dog", "medium_dog"},
        )

    def test_existing_drivers_remain_in_neutral_dog(self) -> None:
        neutral = self.catalog["categories"]["neutral_dog"]
        self.assertEqual(neutral["display_name"], "中性犬")
        self.assertTrue(neutral["runtime_ready"])
        self.assertEqual(set(neutral["actions"]), {"eating", "standing"})
        for action in neutral["actions"].values():
            self.assertIn("/profile_assets/dog_white_model_v1/", action["runtime"])
            self.assertEqual(len(action["runtime_sha256"]), 64)
            self.assertEqual((action["width"], action["height"]), (512, 512))
            self.assertEqual(action["frames"], 81)
            self.assertEqual(action["fps"], 16)
            self.assertFalse(action["audio"])

    def test_six_small_dog_sources_and_test_actions_are_ready(self) -> None:
        small = self.catalog["categories"]["small_dog"]
        self.assertEqual(small["display_name"], "小型犬")
        self.assertFalse(small["runtime_ready"])
        self.assertTrue(small["test_platform_runtime_ready"])
        self.assertEqual(
            small["test_platform_actions"],
            [
                "eating",
                "drinking",
                "running",
                "sleeping",
                "toileting",
                "standing",
            ],
        )
        self.assertEqual(
            small["test_platform_profile_id"],
            "dog_small_white_model_test_v1",
        )
        self.assertEqual(
            set(small["actions"]),
            {
                "sleeping",
                "standing_head_turn",
                "drinking",
                "running",
                "toileting",
                "eating",
            },
        )
        hashes = set()
        for action in small["actions"].values():
            self.assertTrue(action["source"].startswith(self.catalog["asset_root"]))
            self.assertIn("/small_dog/source/", action["source"])
            self.assertEqual(len(action["source_sha256"]), 64)
            hashes.add(action["source_sha256"])
        self.assertEqual(len(hashes), 6)

        for source_action, test_action in (
            ("eating", "eating"),
            ("drinking", "drinking"),
            ("running", "running"),
            ("sleeping", "sleeping"),
            ("toileting", "toileting"),
            ("standing_head_turn", "standing"),
        ):
            action = small["actions"][source_action]
            self.assertEqual(action["test_runtime_action"], test_action)
            self.assertIn("/small_dog/runtime-v2/", action["test_runtime"])
            self.assertIn("/small_dog/runtime-v2/", action["test_reference"])
            self.assertEqual(len(action["test_runtime_sha256"]), 64)
            self.assertEqual(len(action["test_reference_sha256"]), 64)
            self.assertEqual((action["test_width"], action["test_height"]), (512, 512))
            self.assertEqual(action["test_frames"], 81)
            self.assertEqual(action["test_fps"], 16)
            self.assertFalse(action["test_audio"])
            self.assertGreaterEqual(action["test_endpoint_ssim"], 0.9)

    def test_medium_dog_test_profile_aliases_neutral_assets(self) -> None:
        medium = self.catalog["categories"]["medium_dog"]
        self.assertEqual(medium["display_name"], "中型犬")
        self.assertFalse(medium["runtime_ready"])
        self.assertTrue(medium["test_platform_runtime_ready"])
        self.assertEqual(medium["test_platform_actions"], ["eating", "standing"])
        self.assertEqual(
            medium["test_platform_profile_id"],
            "dog_medium_white_model_test_v1",
        )
        self.assertEqual(medium["asset_alias"], "neutral_dog")
        self.assertEqual(medium["actions"], {})


if __name__ == "__main__":
    unittest.main()
