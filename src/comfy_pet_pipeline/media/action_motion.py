"""Local action measurements and anatomy-aware emergency eating deformation."""

from __future__ import annotations

import cv2
import numpy as np


def eating_fallback_motion(rgba: np.ndarray) -> tuple[np.ndarray, dict]:
    """Locate the bowl and low muzzle instead of freezing a fixed body fraction.

    The emergency cycle presses the muzzle toward the food and returns to the
    original pose. It never lifts an occluded muzzle out of a single photograph,
    which would expose missing pixels between the mouth and the fixed bowl.
    """
    height, width = rgba.shape[:2]
    rows, columns = np.indices((height, width))
    rgb = rgba[..., :3].astype(np.int16)
    visible = rgba[..., 3] >= 128
    luma = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)
    neutral = (
        visible & (rows > height * 0.64)
        & (rgb.max(axis=2) - rgb.min(axis=2) < 85)
        & (luma > 110)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(neutral.astype(np.uint8))
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if w >= width * 0.20 and h >= height * 0.055 and y + h >= height * 0.92:
            candidates.append((area, label))
    protected = np.zeros((height, width), dtype=bool)
    policy = 'bounded_upper_head_estimate'
    center_x, center_y = (width - 1) * 0.50, (height - 1) * 0.34
    radius_x, radius_y = max(12.0, width * 0.34), max(12.0, height * 0.30)
    freeze_y = int(round(height * 0.72))
    muzzle = None
    bowl_bbox = None
    if candidates:
        _, label = max(candidates)
        x, y, w, h, _ = (int(v) for v in stats[label])
        bowl_bbox = [x, y, x + w, y + h]
        # The bright connected bowl surface is static, including its rim.
        protected = cv2.dilate((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        search = visible & (columns >= x) & (columns < x + w)
        search &= (rows >= y - max(h, int(height * 0.16))) & (rows < y + h * 0.30)
        dark = search & (luma < 100)
        dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, nose_stats, centroids = cv2.connectedComponentsWithStats(dark)
        noses = []
        for k in range(1, n):
            nx, ny, nw, nh, area = (int(v) for v in nose_stats[k])
            if (area >= 6 and nw < w * 0.70 and nh < height * 0.18
                    and abs(float(centroids[k, 0]) - (x + w / 2)) <= w * 0.20):
                noses.append((-abs(float(centroids[k, 0]) - (x + w / 2)), area, k))
        muzzle = [x + w / 2, y + h * 0.20]
        freeze_y = min(height - 1, round(muzzle[1] + height * 0.025))
        policy = 'bowl_center_contact_cycle'
        if noses:
            _, _, k = max(noses)
            nx, ny, nw, nh, _ = (int(v) for v in nose_stats[k])
            muzzle = [float(centroids[k, 0]), float(centroids[k, 1])]
            freeze_y = min(height - 1, ny + nh + max(6, round(height * 0.04)))
            policy = 'bowl_and_muzzle_anchored_contact_cycle'
        center_x, center_y = muzzle[0], muzzle[1] - height * 0.10
        radius_x, radius_y = max(12.0, w * 0.72), max(12.0, height * 0.24)
    weight = np.exp(-0.5 * (((columns - center_x) / radius_x) ** 2
                          + ((rows - center_y) / radius_y) ** 2)).astype(np.float32)
    weight[rows > freeze_y] = 0.0
    weight[protected] = 0.0
    if bowl_bbox is not None and muzzle is not None:
        x0, _, x1, _ = bowl_bbox
        lateral = (columns < x0) | (columns >= x1)
        fade = np.clip((rows - (muzzle[1] - height * 0.16)) / max(4, height * 0.08), 0, 1)
        weight[lateral] *= (1.0 - fade[lateral]).astype(np.float32)
    return weight, {'policy': policy, 'muzzle_anchor': muzzle, 'bowl_bbox': bowl_bbox,
                    'freeze_below_row': freeze_y, 'head_center': [center_x, center_y],
                    'direction': 'press_toward_food_then_return'}


def _subject_mask(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.int16)
    green = (rgb[..., 1] > rgb[..., 0] + 28) & (rgb[..., 1] > rgb[..., 2] + 28)
    nonwhite = np.min(rgb, axis=2) < 225
    mask = (nonwhite & ~green).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.zeros(mask.shape, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def standing_tail_signature(frames: list[np.ndarray], *, side: int = 1) -> dict:
    """Track the connected tail relative to the upper body, rejecting loose wisps.

    Coordinates are normalized to a 512px subject height and subtract the upper
    body's translation. Whole-frame shake cannot stand in for tail articulation.
    """
    points = []
    for frame in frames:
        mask = _subject_mask(frame)
        ys, xs = np.where(mask)
        if len(xs) < 200:
            continue
        top, bottom = np.percentile(ys, [1, 99])
        extent = max(1.0, bottom - top)
        upper = ys < top + extent * 0.55
        if np.count_nonzero(upper) < 100:
            continue
        anchor_x = float(np.mean(xs[upper]))
        anchor_y = float(np.mean(ys[upper]))
        tail = ((xs - anchor_x) * side > extent * 0.26)
        tail &= (ys > top + extent * 0.48) & (ys < top + extent * 0.90)
        tx, ty = xs[tail], ys[tail]
        if len(tx) < 20:
            continue
        edge = np.percentile(tx * side, 94)
        tip = tx * side >= edge
        points.append([(float(np.median(tx[tip])) - anchor_x) * 512 / extent,
                       (float(np.median(ty[tip])) - anchor_y) * 512 / extent])
    if len(points) < max(5, len(frames) // 2):
        return {'available': False, 'sample_count': len(points), 'span_px': 0.0}
    points_array = np.asarray(points)
    spans = np.percentile(points_array, 90, axis=0) - np.percentile(points_array, 10, axis=0)
    return {'available': True, 'sample_count': len(points),
            'span_px': round(float(np.linalg.norm(spans)), 6),
            'axis_span_px': [round(float(v), 6) for v in spans],
            'points': [[round(float(v), 4) for v in p] for p in points]}


def _head_palette(frame: np.ndarray) -> dict:
    mask = _subject_mask(frame)
    ys, _ = np.where(mask)
    if len(ys) < 200:
        return {'available': False}
    top, bottom = np.percentile(ys, [1, 99])
    rows = np.indices(mask.shape)[0]
    pixels = frame[mask & (rows < top + (bottom - top) * 0.40)].astype(np.float32)
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    return {'available': bool(len(pixels)),
            'subject_height_fraction': float((bottom - top) / frame.shape[0]),
            'chromatic_fraction': float(np.mean(chroma > 35)),
            'dark_fraction': float(np.mean(pixels.mean(axis=1) < 95))}


def standing_motion_quality(
    frames: list[np.ndarray], driver_frames: list[np.ndarray],
    *, reference_frame: np.ndarray | None = None,
) -> dict:
    driver = standing_tail_signature(driver_frames)
    output = standing_tail_signature(frames)
    required = max(12.0, min(36.0, float(driver['span_px']) * 0.35))
    ratio = float(output['span_px']) / required
    issues = []
    if not driver['available']:
        issues.append('standing_driver_tail_motion_unavailable')
    if not output['available']:
        issues.append('standing_connected_tail_motion_unavailable')
    elif ratio < 1.0:
        issues.append('standing_tail_motion_too_low')
    identity = {'available': False, 'acceptable': reference_frame is None}
    if reference_frame is not None:
        reference = _head_palette(reference_frame)
        outputs = [_head_palette(frames[i]) for i in sorted({0, len(frames)//2, len(frames)-1})] if frames else []
        valid = [p for p in outputs if p['available']]
        identity = {'available': reference['available'] and len(valid) == 3,
                    'acceptable': True, 'reference': reference, 'output_samples': outputs}
        if not identity['available']:
            issues.append('standing_head_identity_unavailable')
            identity['acceptable'] = False
        else:
            if min(p['subject_height_fraction'] for p in valid) < reference['subject_height_fraction'] * 0.80:
                issues.append('standing_head_silhouette_unavailable')
                identity['acceptable'] = False
            for key, floor in [('chromatic_fraction', 0.20), ('dark_fraction', 0.18)]:
                if reference[key] >= floor and min(p[key] for p in valid) < reference[key] * 0.60:
                    issues.append('standing_reference_head_' + key + '_collapsed')
                    identity['acceptable'] = False
    return {'policy': 'standing_connected_tail_motion_relative_to_body_v1',
            'available': bool(driver['available'] and output['available']),
            'acceptable': not issues, 'issues': issues, 'blocking': False,
            'output_tail': output, 'driver_tail': driver,
            'required_span_px': round(required, 6), 'motion_ratio': round(ratio, 6),
            'head_identity': identity}
