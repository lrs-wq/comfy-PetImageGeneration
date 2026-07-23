from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "cat_poses_clean_alpha" / "transparent" / "pose_02_clean_alpha.png"
OUTPUT_DIR = ROOT / "outputs" / "cat_poses_no_whiskers" / "transparent"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.open(INPUT).convert("RGBA")
    rgba = np.array(image)
    rgb = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = rgba[:, :, 3]

    mask = np.zeros(alpha.shape, dtype=np.uint8)

    # Manual whisker strokes for pose_02. The alpha is preserved; only the RGB
    # whisker lines are inpainted so Hunyuan does not turn them into geometry.
    strokes = [
        ((292, 452), (125, 398), 13),
        ((300, 466), (120, 458), 13),
        ((304, 482), (135, 520), 13),
        ((308, 500), (165, 580), 12),
        ((337, 450), (178, 410), 12),
        ((342, 466), (175, 462), 12),
        ((344, 486), (190, 535), 12),
        ((366, 452), (575, 398), 13),
        ((370, 468), (590, 458), 13),
        ((371, 486), (565, 535), 13),
        ((372, 503), (525, 585), 12),
        ((338, 450), (515, 410), 12),
        ((340, 466), (520, 462), 12),
        ((341, 486), (510, 535), 12),
    ]
    for start, end, width in strokes:
        cv2.line(mask, start, end, 255, max(6, width - 3), lineType=cv2.LINE_AA)

    # Keep edits constrained to the face/muzzle area and inside the cat alpha.
    face_roi = np.zeros_like(mask)
    cv2.ellipse(face_roi, (335, 470), (250, 170), 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, face_roi)
    mask[alpha < 20] = 0
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    feather = cv2.GaussianBlur(mask, (35, 35), 0).astype(np.float32) / 255.0
    feather = np.clip(feather * 0.82, 0.0, 0.82)[:, :, None]

    soft = cv2.bilateralFilter(rgb, 19, 46, 46)
    soft = cv2.GaussianBlur(soft, (9, 9), 0)
    blended = (rgb.astype(np.float32) * (1.0 - feather) + soft.astype(np.float32) * feather)
    out_rgb = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    out = np.dstack([out_rgb, alpha])

    output = OUTPUT_DIR / "pose_02_no_whiskers.png"
    Image.fromarray(out).save(output)

    compare = Image.new("RGB", (image.width * 2, image.height), "white")
    compare.paste(image.convert("RGB"), (0, 0))
    compare.paste(Image.fromarray(out).convert("RGB"), (image.width, 0))
    compare.save(OUTPUT_DIR / "pose_02_no_whiskers_compare.jpg", quality=95)
    print(output)


if __name__ == "__main__":
    main()
