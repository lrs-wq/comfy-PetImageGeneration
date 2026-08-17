#!/usr/bin/env python3
"""Force uploaded images into the COCO cat/dog classes with YOLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


CLASS_IDS = {"cat": 15, "dog": 16}
POLICY = "yolo_coco_cat_dog_forced_argmax_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.0001)
    return parser.parse_args()


def maximum_scores(result: Any) -> dict[str, float]:
    scores = {species: 0.0 for species in CLASS_IDS}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return scores
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    for class_id, confidence in zip(classes, confidences, strict=True):
        for species, expected_class_id in CLASS_IDS.items():
            if int(class_id) == expected_class_id:
                scores[species] = max(scores[species], float(confidence))
    return scores


def main() -> int:
    args = parse_args()
    model = YOLO(str(args.model))
    results = model.predict(
        source=[str(path) for path in args.image],
        classes=list(CLASS_IDS.values()),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        stream=False,
    )
    per_view = [maximum_scores(result) for result in results]
    if len(per_view) != len(args.image):
        raise RuntimeError("Classifier did not return one result per input view")
    denominator = float(len(per_view))
    scores = {
        species: sum(item[species] for item in per_view) / denominator
        for species in CLASS_IDS
    }
    # Ties, including a complete absence of detections, intentionally choose cat.
    species = "dog" if scores["dog"] > scores["cat"] else "cat"
    payload = {
        "species": species,
        "scores": scores,
        "margin": abs(scores["dog"] - scores["cat"]),
        "per_view": per_view,
        "view_count": len(per_view),
        "policy": POLICY,
        "forced_binary": True,
        "unknown_class": False,
        "model": str(args.model),
        "class_ids": CLASS_IDS,
        "confidence_floor": args.conf,
        "image_size": args.imgsz,
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
