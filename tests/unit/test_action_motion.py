from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.comfy_pet_pipeline.media.action_motion import (
    eating_fallback_motion, standing_motion_quality,
)
from src.comfy_pet_pipeline.api.local_video_parts.subject_replacement import (
    eating_reference_motion_recovery_required, standing_scail_best_effort_score,
)


def standing_frames(*, moving=True, translated=False, ghost=False, white_head=False):
    frames = []
    for i in range(25):
        phase = 2 * np.pi * i / 24
        frame = np.full((512, 512, 3), 255, np.uint8)
        coat = (192, 124, 58)
        cv2.ellipse(frame, (225, 290), (68, 140), 0, 0, 360, coat, -1)
        cv2.circle(frame, (200, 130), 60, (225, 225, 225) if white_head else coat, -1)
        tip = (int(390 + 65 * np.sin(phase)), int(345 + 35 * np.cos(phase))) if moving else (385, 345)
        cv2.line(frame, (270, 355), tip, coat, 22)
        if ghost:
            cv2.circle(frame, (int(465 + 25*np.sin(phase)), int(310 + 40*np.cos(phase))), 4, coat, -1)
        if translated:
            frame = cv2.warpAffine(frame, np.float32([[1, 0, round(9*np.sin(phase))], [0, 1, round(7*np.cos(phase))]]), (512, 512), borderValue=(255,255,255))
        frames.append(frame)
    return frames


def test_standing_requires_articulated_tail_not_static_or_camera_shake():
    driver = standing_frames()
    for frames in [standing_frames(moving=False), standing_frames(moving=False, translated=True), standing_frames(moving=False, ghost=True)]:
        quality = standing_motion_quality(frames, driver)
        assert not quality['acceptable'], quality
        assert 'standing_tail_motion_too_low' in quality['issues']
    assert standing_motion_quality(driver, driver)['acceptable']


def test_standing_moving_white_driver_head_does_not_replace_golden_identity():
    driver = standing_frames()
    quality = standing_motion_quality(standing_frames(white_head=True), driver, reference_frame=driver[0])
    assert quality['motion_ratio'] >= 1
    assert not quality['acceptable']
    assert not quality['head_identity']['acceptable']
    assert standing_motion_quality(driver, driver, reference_frame=driver[0])['acceptable']


def test_standing_best_effort_prefers_identity_safe_actual_video():
    unsafe = {'motion_ratio': 20, 'head_identity': {'acceptable': False}}
    safe = {'motion_ratio': 0.8, 'head_identity': {'acceptable': True}}
    assert standing_scail_best_effort_score(safe)['score'] > standing_scail_best_effort_score(unsafe)['score']


def test_low_muzzle_below_old_freeze_line_moves_while_bowl_and_paw_stay_fixed():
    rgba = np.zeros((420,340,4), np.uint8)
    coat = (200,135,60,255)
    cv2.ellipse(rgba,(195,145),(120,110),0,0,360,coat,-1)
    cv2.rectangle(rgba,(180,220),(215,393),coat,-1)
    cv2.ellipse(rgba,(100,287),(48,75),0,0,360,coat,-1)
    cv2.ellipse(rgba,(103,349),(10,6),0,0,360,(35,28,22,255),-1)
    cv2.rectangle(rgba,(48,368),(155,414),(195,184,163,255),-1)
    weight, metadata = eating_fallback_motion(rgba)
    assert metadata['bowl_bbox'] is not None
    assert weight[349,103] > 0.5  # This is below the old 72% freeze boundary.
    y,x = np.indices(weight.shape)
    down = cv2.remap(rgba,x.astype(np.float32),(y-12*weight).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    assert np.abs(down[335:366,85:120].astype(int)-rgba[335:366,85:120]).max() > 30
    np.testing.assert_array_equal(down[370:414,50:153],rgba[370:414,50:153])
    np.testing.assert_array_equal(down[345:393,180:215],rgba[345:393,180:215])


def test_reference_motion_failure_can_reprepare_but_integrity_failure_cannot():
    metadata = {'policy':'reference_space_head_nod_food_overlap_v2'}
    issues = ['eating_muzzle_motion_too_low']
    assert eating_reference_motion_recovery_required('eating','dog',metadata,issues)
    assert not eating_reference_motion_recovery_required('eating','cat',metadata,issues)
    assert not eating_reference_motion_recovery_required('eating','dog',metadata,issues+['source_digest_changed'])
    assert not eating_reference_motion_recovery_required('eating','dog',{'full_white_driver_geometry':True},issues)


@pytest.mark.parametrize('controls,unsafe', [([-12, 0, -12], False), ([-13.558, 22.442, -13.558], True)])
def test_prepared_muzzle_motion_pass_does_not_hide_upward_bowl_contact_tear(monkeypatch, controls, unsafe):
    from pathlib import Path
    from src.comfy_pet_pipeline.api import local_video as api
    from src.comfy_pet_pipeline.api.local_video_parts import subject_replacement as replacement

    monkeypatch.setattr(api, 'evaluate_eating_scail_video_motion', lambda *args, **kwargs: {
        'available': True, 'acceptable': True, 'issues': [],
    })
    monkeypatch.setattr(replacement, 'evaluate_eating_scail_video_motion', api.evaluate_eating_scail_video_motion)
    metadata = {'policy': 'reference_space_head_nod_food_overlap_v2', 'vertical_control_px_by_frame': controls}
    quality = replacement.evaluate_eating_prepared_pose_motion(Path('unused.mp4'), pose_metadata=metadata)
    assert quality['acceptable'] is not unsafe
    assert eating_reference_motion_recovery_required('eating', 'dog', metadata, quality['issues']) is unsafe
    if unsafe:
        assert quality['issues'] == ['eating_reference_upward_contact_warp_unsafe']
        assert quality['upward_contact_warp_px_max'] == 22.442
        assert not eating_reference_motion_recovery_required('eating', 'dog', metadata, quality['issues'] + ['source_digest_changed'])
