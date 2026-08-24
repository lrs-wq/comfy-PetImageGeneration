import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATHS = {
    "medium": PROJECT_ROOT / "profiles" / "dog_medium_white_model_test_v1.json",
    "small": PROJECT_ROOT / "profiles" / "dog_small_white_model_test_v1.json",
}


class DogSizeTestProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = {
            size: json.loads(path.read_text(encoding="utf-8"))
            for size, path in PROFILE_PATHS.items()
        }

    def test_profiles_are_explicitly_test_only_dog_profiles(self) -> None:
        expected_ids = {
            "medium": "dog_medium_white_model_test_v1",
            "small": "dog_small_white_model_test_v1",
        }
        for size, profile in self.profiles.items():
            self.assertEqual(profile["schema_version"], 2)
            self.assertEqual(profile["id"], expected_ids[size])
            self.assertEqual(profile["species"], "dog")
            self.assertEqual(profile["supported_actions"], ["eating", "standing"])
            self.assertEqual(profile["status"], "test_platform_only")
            self.assertTrue(profile["test_platform_only"])
            self.assertEqual(profile["morphology_category"], f"{size}_dog")

    def test_medium_profile_reuses_existing_neutral_dog_assets(self) -> None:
        profile = self.profiles["medium"]
        self.assertEqual(profile["asset_alias"], "neutral_dog")
        runtime_actions = profile["runtime"]["actions"]
        for action in ("eating", "standing"):
            self.assertIn(
                "/profile_assets/dog_white_model_v1/",
                runtime_actions[action]["driving_animation"],
            )
            self.assertEqual(len(runtime_actions[action]["driving_animation_sha256"]), 64)

    def test_small_profile_uses_normalized_runtime_and_frozen_references(self) -> None:
        profile = self.profiles["small"]
        self.assertEqual(profile["runtime"]["frames"], 81)
        self.assertEqual(profile["runtime"]["fps"], 16)
        normalization = profile["uploaded_identity_driving"]["normalization"]
        self.assertEqual(normalization["size"], [512, 512])
        self.assertTrue(normalization["audio_removed"])
        for action in ("eating", "standing"):
            driving = profile["uploaded_identity_driving"]["actions"][action]
            runtime = profile["runtime"]["actions"][action]
            self.assertIn("/small_dog/runtime-v1/", driving["runtime"])
            self.assertIn("/small_dog/runtime-v1/", driving["geometry_reference"])
            self.assertEqual(runtime["driving_animation"], driving["runtime"])
            self.assertEqual(
                runtime["driving_animation_sha256"],
                driving["runtime_sha256"],
            )
            self.assertEqual(len(driving["runtime_sha256"]), 64)
            self.assertEqual(len(driving["geometry_reference_sha256"]), 64)
            self.assertGreaterEqual(driving["endpoint_ssim"], 0.9)


if __name__ == "__main__":
    unittest.main()
