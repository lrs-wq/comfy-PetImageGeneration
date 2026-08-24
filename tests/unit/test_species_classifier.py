import io
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from src.comfy_pet_pipeline.identity import species_classifier


def png_bytes(color=(120, 90, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


class SpeciesClassifierTests(unittest.TestCase):
    def dependencies(self, root: Path) -> tuple[Path, Path, Path]:
        python = root / "python"
        model = root / "model.pt"
        runner = root / "tools" / "classification" / "classify_pet_species.py"
        runner.parent.mkdir(parents=True)
        for path in (python, model, runner):
            path.touch()
        return python, model, runner

    def test_dog_result_is_preserved_as_forced_binary_decision(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            python, model, _ = self.dependencies(root)
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "model diagnostic\n"
                    '{"species":"dog","scores":{"cat":0.01,"dog":0.82},'
                    '"margin":0.81,"per_view":[],"view_count":1}'
                ),
                stderr="",
            )
            with patch.object(
                species_classifier.subprocess,
                "run",
                return_value=completed,
            ):
                result = species_classifier.classify_pet_species(
                    {"front": png_bytes()},
                    project_root=root,
                    python=python,
                    model=model,
                )

        self.assertEqual(result["species"], "dog")
        self.assertTrue(result["forced_binary"])
        self.assertFalse(result["unknown_class"])
        self.assertFalse(result["fallback"])

    def test_classifier_failure_accepts_input_and_falls_back_to_cat(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            python, model, _ = self.dependencies(root)
            with patch.object(
                species_classifier.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("classifier", 1),
            ):
                result = species_classifier.classify_pet_species(
                    {"front": png_bytes((0, 0, 255))},
                    project_root=root,
                    python=python,
                    model=model,
                    timeout=5,
                )

        self.assertEqual(result["species"], "cat")
        self.assertTrue(result["forced_binary"])
        self.assertFalse(result["unknown_class"])
        self.assertTrue(result["fallback"])
        self.assertIn("TimeoutExpired", result["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
