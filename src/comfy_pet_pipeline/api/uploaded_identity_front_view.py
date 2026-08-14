"""Advisory front-view audit for uploaded pet identity photographs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

LOGGER = logging.getLogger(__name__)

UPLOADED_FRONT_VIEW_AUDIT_POLICY = "opencv_frontal_cat_face_advisory_v1"
FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE = "front_view_not_reliably_frontal"
UPLOADED_FRONT_VIEW_AUDIT_MAX_DIMENSION = 1280
UPLOADED_FRONT_VIEW_AUDIT_MIN_FACE_AREA_FRACTION = 0.0125
UPLOADED_FRONT_VIEW_AUDIT_MIN_FACE_WIDTH_FRACTION = 0.10
UPLOADED_FRONT_VIEW_AUDIT_CASCADES = (
    "haarcascade_frontalcatface.xml",
    "haarcascade_frontalcatface_extended.xml",
)


def _audit_uploaded_front_identity_view(image_path: Path) -> dict[str, Any]:
    """Detect a plausible frontal cat face without blocking generation."""

    audit: dict[str, Any] = {
        "policy": UPLOADED_FRONT_VIEW_AUDIT_POLICY,
        "role": "front",
        "blocking": False,
        "acceptable": False,
        "status": "unavailable_advisory",
        "issues": [],
        "image_path": str(image_path),
        "classifiers_requested": list(UPLOADED_FRONT_VIEW_AUDIT_CASCADES),
        "classifiers_loaded": [],
        "detections": [],
        "qualified_detections": [],
        "thresholds": {
            "min_face_area_fraction": (
                UPLOADED_FRONT_VIEW_AUDIT_MIN_FACE_AREA_FRACTION
            ),
            "min_face_width_fraction": (
                UPLOADED_FRONT_VIEW_AUDIT_MIN_FACE_WIDTH_FRACTION
            ),
            "max_audit_dimension": UPLOADED_FRONT_VIEW_AUDIT_MAX_DIMENSION,
            "scale_factor": 1.05,
            "min_neighbors": 3,
        },
    }
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        audit["issues"] = ["front_view_audit_image_unreadable"]
        return audit

    original_height, original_width = image.shape[:2]
    scale = min(
        1.0,
        UPLOADED_FRONT_VIEW_AUDIT_MAX_DIMENSION
        / float(max(original_height, original_width)),
    )
    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    height, width = gray.shape[:2]
    min_face_size = max(32, int(round(min(height, width) * 0.08)))
    cascade_root = Path(cv2.data.haarcascades)
    detections: list[dict[str, Any]] = []
    loaded: list[str] = []
    for cascade_name in UPLOADED_FRONT_VIEW_AUDIT_CASCADES:
        cascade_path = cascade_root / cascade_name
        if not cascade_path.is_file():
            continue
        try:
            classifier = cv2.CascadeClassifier(str(cascade_path))
            if classifier.empty():
                continue
            loaded.append(cascade_name)
            boxes = classifier.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(min_face_size, min_face_size),
            )
        except cv2.error:
            LOGGER.exception(
                "Uploaded front-view classifier %s failed for %s",
                cascade_name,
                image_path,
            )
            continue
        for raw_x, raw_y, raw_width, raw_height in boxes:
            x = int(raw_x)
            y = int(raw_y)
            face_width = int(raw_width)
            face_height = int(raw_height)
            area_fraction = (face_width * face_height) / float(width * height)
            width_fraction = face_width / float(min(width, height))
            center_x_fraction = (x + face_width / 2.0) / float(width)
            center_y_fraction = (y + face_height / 2.0) / float(height)
            qualified = bool(
                area_fraction
                >= UPLOADED_FRONT_VIEW_AUDIT_MIN_FACE_AREA_FRACTION
                and width_fraction
                >= UPLOADED_FRONT_VIEW_AUDIT_MIN_FACE_WIDTH_FRACTION
                and 0.08 <= center_x_fraction <= 0.92
                and 0.03 <= center_y_fraction <= 0.80
            )
            detections.append(
                {
                    "classifier": cascade_name,
                    "x": x,
                    "y": y,
                    "width": face_width,
                    "height": face_height,
                    "area_fraction": round(area_fraction, 6),
                    "width_fraction": round(width_fraction, 6),
                    "center_x_fraction": round(center_x_fraction, 6),
                    "center_y_fraction": round(center_y_fraction, 6),
                    "qualified": qualified,
                }
            )

    audit.update(
        {
            "image_width": original_width,
            "image_height": original_height,
            "audit_width": width,
            "audit_height": height,
            "classifiers_loaded": loaded,
            "detections": detections,
            "qualified_detections": [
                detection for detection in detections if detection["qualified"]
            ],
        }
    )
    if not loaded:
        audit["issues"] = ["front_view_audit_classifier_unavailable"]
        return audit
    if audit["qualified_detections"]:
        audit["acceptable"] = True
        audit["status"] = "passed_advisory"
        return audit

    audit["status"] = "warning_advisory"
    audit["issues"] = [FRONT_VIEW_NOT_RELIABLY_FRONTAL_STAGE]
    return audit


def audit_uploaded_front_identity_view(image_path: Path) -> dict[str, Any]:
    """Run the advisory detector and fail open on audit infrastructure errors."""

    try:
        return _audit_uploaded_front_identity_view(image_path)
    except Exception:  # noqa: BLE001 - this audit is deliberately fail-open
        LOGGER.exception("Uploaded front-view audit failed for %s", image_path)
        return {
            "policy": UPLOADED_FRONT_VIEW_AUDIT_POLICY,
            "role": "front",
            "blocking": False,
            "acceptable": False,
            "status": "unavailable_advisory",
            "issues": ["front_view_audit_internal_error"],
            "image_path": str(image_path),
            "classifiers_requested": list(UPLOADED_FRONT_VIEW_AUDIT_CASCADES),
            "classifiers_loaded": [],
            "detections": [],
            "qualified_detections": [],
        }
