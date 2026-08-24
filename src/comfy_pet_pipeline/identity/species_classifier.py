"""Isolated cat/dog classification with an accept-all binary contract."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError


POLICY = "yolo_coco_cat_dog_forced_argmax_v1"
FALLBACK_POLICY = "classifier_unavailable_defaults_to_cat_v1"
SUPPORTED_SPECIES = ("cat", "dog")


def _last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Classifier did not emit a JSON object")


def _fallback(reason: str, *, view_count: int) -> dict[str, Any]:
    return {
        "species": "cat",
        "scores": {"cat": 0.0, "dog": 0.0},
        "margin": 0.0,
        "per_view": [],
        "view_count": view_count,
        "policy": FALLBACK_POLICY,
        "classifier_policy": POLICY,
        "forced_binary": True,
        "unknown_class": False,
        "fallback": True,
        "fallback_reason": reason[:500],
    }


def classify_pet_species(
    image_views: Mapping[str, bytes],
    *,
    project_root: Path,
    python: Path | None = None,
    model: Path | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Return cat or dog for every input, falling back to cat on any failure."""

    if not image_views:
        return _fallback("no_uploaded_identity_views", view_count=0)
    classifier_python = Path(
        python
        or os.environ.get(
            "PET_SPECIES_CLASSIFIER_PYTHON",
            "/opt/miniconda3/envs/yolo/bin/python",
        )
    ).expanduser()
    classifier_model = Path(
        model
        or os.environ.get("PET_SPECIES_CLASSIFIER_MODEL", "/root/yolo26n.pt")
    ).expanduser()
    runner = project_root / "tools" / "classification" / "classify_pet_species.py"
    classifier_timeout = max(
        5,
        min(
            300,
            int(timeout or os.environ.get("PET_SPECIES_CLASSIFIER_TIMEOUT", "90")),
        ),
    )
    missing = [
        str(path)
        for path in (classifier_python, classifier_model, runner)
        if not path.is_file()
    ]
    if missing:
        return _fallback(
            "missing_classifier_dependency:" + ",".join(missing),
            view_count=len(image_views),
        )

    try:
        with tempfile.TemporaryDirectory(prefix="pet-species-") as temporary:
            input_paths: list[Path] = []
            for index, (role, raw) in enumerate(image_views.items()):
                try:
                    with Image.open(io.BytesIO(raw)) as loaded:
                        normalized = loaded.convert("RGB")
                except (OSError, UnidentifiedImageError) as exc:
                    raise ValueError(f"invalid_{role}_image:{exc}") from exc
                path = Path(temporary) / f"{index:02d}-{role}.png"
                normalized.save(path, format="PNG", optimize=True)
                input_paths.append(path)
            command = [
                str(classifier_python),
                str(runner),
                "--model",
                str(classifier_model),
            ]
            for path in input_paths:
                command.extend(["--image", str(path)])
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=classifier_timeout,
            )
        payload = _last_json_object(completed.stdout)
        species = str(payload.get("species") or "").strip().lower()
        if species not in SUPPORTED_SPECIES:
            raise ValueError(f"unsupported_classifier_species:{species!r}")
        payload["species"] = species
        payload["forced_binary"] = True
        payload["unknown_class"] = False
        payload["fallback"] = False
        payload["python"] = str(classifier_python)
        payload["model"] = str(classifier_model)
        return payload
    except Exception as exc:  # noqa: BLE001 - acceptance must never depend on ML health
        return _fallback(
            f"{type(exc).__name__}:{exc}",
            view_count=len(image_views),
        )
