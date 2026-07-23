from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import dashscope
import requests
from dashscope import VideoSynthesis
from PIL import Image

from pet_pose_generator import call_qwen_image_edit, download_image, extract_image_urls, image_to_data_url, read_api_key


EATING_KEYFRAME_PROMPT = """Use the input image as the identity and style reference for the same cat.
Create a clean first frame for image-to-video generation.
The same cute 3D plush cartoon cat is eating from a small blue ceramic cat food bowl in a three-quarter side view.
The camera sees the cat from a slightly front-side angle, about 35 to 45 degrees from the front, not a straight front view.
The bowl is placed just in front of the cat's mouth and slightly toward the visible side, containing visible brown dry cat food kibbles.
The cat's head is already lowered close to the food bowl, mouth near the food, ready to chew, with the body angled gently sideways.
Keep the same fluffy long fur, white chest, gray-brown tabby forehead and ears, soft gray facial mask, pink nose, and glossy eyes.
One cat only, full body visible, centered, fixed camera, clean pure white studio background, no text, no watermark, no extra animals.
Make the frame stable and suitable as the first frame of a 5 second eating animation."""

EATING_KEYFRAME_NEGATIVE_PROMPT = """multiple cats, extra animal, human, hand, text, watermark, logo, complex background, furniture, cropped body, cropped ears, cropped tail, no bowl, empty bowl, water bowl, deformed face, extra legs, blurry, low quality"""

EATING_VIDEO_PROMPT = """A cute 3D plush cartoon cat eats dry cat food from a small blue ceramic bowl.
The cat is shown in a three-quarter side view, about 35 to 45 degrees from the front, so the body and bowl are visibly angled sideways.
The cat gently lowers and raises its head in small motions, softly chewing the food.
Ears twitch slightly, whiskers move subtly, and the tail moves gently.
The cat remains the same character with consistent fur color, face, body proportions, and plush cartoon style.
Fixed camera, clean white studio background, full body stays in frame, no scene cuts, no camera zoom, no extra animals, no humans."""

EATING_VIDEO_NEGATIVE_PROMPT = """scene cut, camera movement, zoom, pan, rotation, extra cat, extra animal, human, hand, text, watermark, changing fur color, deformed face, deformed legs, cropped body, leaving frame, complex background, eating without bowl, water bowl"""

RUNNING_KEYFRAME_PROMPT = """Use the input image as the identity and style reference for the same cat.
Create a clean first frame for image-to-video generation.
The same cute 3D plush cartoon cat is running in a lively side-view pose, as if moving from left to right.
Keep the full body visible, with four legs in a clear natural running stride, tail balanced behind the body, and the head facing forward.
Preserve the same fluffy long fur, mostly white chest and body, gray-brown tabby forehead and ears, soft gray facial mask, pink nose, glossy eyes, and cute plush 3D cartoon style.
One cat only, no props, no text, no watermark, no extra animals.
Fixed camera, clean pure white studio background, centered composition, stable frame for a 5 second running animation."""

RUNNING_KEYFRAME_NEGATIVE_PROMPT = """multiple cats, extra animal, human, hand, text, watermark, logo, complex background, furniture, food bowl, collar change, cropped body, cropped ears, cropped tail, deformed face, deformed legs, extra legs, missing legs, blurry, low quality"""

RUNNING_VIDEO_PROMPT = """A cute 3D plush cartoon cat runs from left to right in place on a clean white studio background.
The cat keeps the same character identity, fur color, face, body proportions, and soft plush cartoon style.
Four legs alternate naturally in a smooth running cycle, the tail sways gently for balance, ears and whiskers move subtly.
Fixed camera, full body stays centered and fully visible, no scene cuts, no camera zoom, no extra animals, no humans."""

RUNNING_VIDEO_NEGATIVE_PROMPT = """scene cut, camera movement, zoom, pan, rotation, extra cat, extra animal, human, hand, text, watermark, changing fur color, deformed face, deformed legs, extra legs, missing legs, cropped body, leaving frame, complex background, food bowl, sliding without leg motion"""


@dataclass(frozen=True)
class EatingVideoResult:
    keyframe_path: Path
    raw_video_path: Path
    final_video_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class RunningVideoResult:
    keyframe_path: Path
    raw_video_path: Path
    final_video_path: Path
    metadata_path: Path


def ensure_rgb_png(image_path: Path, output_path: Path, canvas_size: tuple[int, int] = (1280, 720)) -> Path:
    image = Image.open(image_path).convert("RGBA")
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    max_w, max_h = int(canvas_size[0] * 0.78), int(canvas_size[1] * 0.86)
    image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (canvas_size[0] - image.width) // 2
    y = (canvas_size[1] - image.height) // 2
    canvas.alpha_composite(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    return output_path


def generate_eating_keyframe(
    *,
    pose_image_path: Path,
    api_key: str,
    output_dir: Path,
    model: str,
    seed: int,
    endpoint: str,
) -> Path:
    prepared_input = ensure_rgb_png(
        pose_image_path,
        output_dir / "input" / "pose2_white_16x9.png",
    )
    response = call_qwen_image_edit(
        image_path=prepared_input,
        api_key=api_key,
        prompt=EATING_KEYFRAME_PROMPT,
        negative_prompt=EATING_KEYFRAME_NEGATIVE_PROMPT,
        output_count=1,
        size="1280*720",
        model=model,
        seed=seed,
        endpoint=endpoint,
    )
    url = extract_image_urls(response)[0]
    keyframe_path = download_image(url, output_dir / "keyframes" / "eat_keyframe_000.png")
    Image.open(keyframe_path).convert("RGB").save(keyframe_path)
    return keyframe_path


def generate_running_keyframe(
    *,
    pose_image_path: Path,
    api_key: str,
    output_dir: Path,
    model: str,
    seed: int,
    endpoint: str,
) -> Path:
    prepared_input = ensure_rgb_png(
        pose_image_path,
        output_dir / "input" / "pose2_white_16x9.png",
    )
    response = call_qwen_image_edit(
        image_path=prepared_input,
        api_key=api_key,
        prompt=RUNNING_KEYFRAME_PROMPT,
        negative_prompt=RUNNING_KEYFRAME_NEGATIVE_PROMPT,
        output_count=1,
        size="1280*720",
        model=model,
        seed=seed,
        endpoint=endpoint,
    )
    url = extract_image_urls(response)[0]
    keyframe_path = download_image(url, output_dir / "keyframes" / "run_keyframe_000.png")
    Image.open(keyframe_path).convert("RGB").save(keyframe_path)
    return keyframe_path


def submit_and_wait_video(
    *,
    keyframe_path: Path,
    api_key: str,
    output_dir: Path,
    model: str,
    resolution: str,
    duration: int,
    seed: int,
    endpoint: str,
    poll_interval: int,
    timeout_seconds: int,
    prompt: str = EATING_VIDEO_PROMPT,
    negative_prompt: str = EATING_VIDEO_NEGATIVE_PROMPT,
    action_name: str = "eat",
) -> tuple[Path, object]:
    if model.startswith("wanx2.1-") or model.startswith("wan2.2-"):
        return submit_and_wait_video_legacy(
            keyframe_path=keyframe_path,
            api_key=api_key,
            output_dir=output_dir,
            model=model,
            resolution=resolution,
            duration=duration,
            seed=seed,
            endpoint=endpoint,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
            prompt=prompt,
            negative_prompt=negative_prompt,
            action_name=action_name,
        )

    dashscope.base_http_api_url = endpoint
    response = VideoSynthesis.async_call(
        api_key=api_key,
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        media=[{"type": "first_frame", "url": str(keyframe_path)}],
        resolution=resolution,
        duration=duration,
        prompt_extend=False,
        watermark=False,
        seed=seed,
    )
    if getattr(response, "status_code", None) != 200:
        raise RuntimeError(
            f"Wan video submit failed: status={getattr(response, 'status_code', None)} "
            f"code={getattr(response, 'code', None)} message={getattr(response, 'message', None)}"
        )

    task_id = response.output.task_id
    deadline = time.time() + timeout_seconds
    result = response
    while time.time() < deadline:
        result = VideoSynthesis.fetch(task_id, api_key=api_key)
        status = result.output.task_status
        if status == "SUCCEEDED":
            break
        if status in {"FAILED", "CANCELED", "UNKNOWN"}:
            raise RuntimeError(f"Wan video task {task_id} ended with {status}: {result}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Wan video task {task_id} did not finish within {timeout_seconds} seconds.")

    video_url = result.output.video_url
    if not video_url:
        raise RuntimeError(f"Wan video task {task_id} succeeded but no video_url was returned.")

    raw_video_path = output_dir / "raw_videos" / f"{action_name}_raw.mp4"
    raw_video_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(video_url, timeout=180) as video_response:
        video_response.raise_for_status()
        raw_video_path.write_bytes(video_response.content)
    return raw_video_path, result


def submit_and_wait_video_legacy(
    *,
    keyframe_path: Path,
    api_key: str,
    output_dir: Path,
    model: str,
    resolution: str,
    duration: int,
    seed: int,
    endpoint: str,
    poll_interval: int,
    timeout_seconds: int,
    prompt: str,
    negative_prompt: str,
    action_name: str,
) -> tuple[Path, dict]:
    base_endpoint = endpoint.rstrip("/")
    submit_url = f"{base_endpoint}/services/aigc/video-generation/video-synthesis"
    payload = {
        "model": model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "img_url": image_to_data_url(keyframe_path),
        },
        "parameters": {
            "resolution": resolution,
            "duration": duration,
            "prompt_extend": False,
            "watermark": False,
            "seed": seed,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    response = requests.post(submit_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"DashScope legacy video submit failed: status={response.status_code} body={response.text}")
    submit_result = response.json()
    task_id = submit_result.get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"DashScope legacy video submit returned no task_id: {submit_result}")

    deadline = time.time() + timeout_seconds
    result = submit_result
    while time.time() < deadline:
        poll_response = requests.get(
            f"{base_endpoint}/tasks/{task_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        if poll_response.status_code != 200:
            raise RuntimeError(
                f"DashScope legacy video poll failed: status={poll_response.status_code} body={poll_response.text}"
            )
        result = poll_response.json()
        status = result.get("output", {}).get("task_status")
        if status == "SUCCEEDED":
            break
        if status in {"FAILED", "CANCELED", "UNKNOWN"}:
            raise RuntimeError(f"DashScope legacy video task {task_id} ended with {status}: {result}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"DashScope legacy video task {task_id} did not finish within {timeout_seconds} seconds.")

    video_url = result.get("output", {}).get("video_url")
    if not video_url:
        raise RuntimeError(f"DashScope legacy video task {task_id} succeeded but no video_url was returned.")

    raw_video_path = output_dir / "raw_videos" / f"{action_name}_raw.mp4"
    raw_video_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(video_url, timeout=180) as video_response:
        video_response.raise_for_status()
        raw_video_path.write_bytes(video_response.content)
    return raw_video_path, result


def postprocess_video(raw_video_path: Path, final_video_path: Path, ffmpeg: str = "ffmpeg") -> Path:
    final_video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_video_path),
        "-vf",
        "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,fps=30,trim=end_frame=150,setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_video_path),
    ]
    subprocess.run(command, check=True)
    return final_video_path


def get_video_response_field(video_response: object, field: str):
    if isinstance(video_response, dict):
        return video_response.get("output", {}).get(field)
    output = getattr(video_response, "output", None)
    return getattr(output, field, None)


def get_video_response_usage(video_response: object):
    if isinstance(video_response, dict):
        return video_response.get("usage")
    return getattr(video_response, "usage", None)


def generate_eating_video(
    *,
    pose_image_path: Path,
    key_file: Path,
    output_dir: Path,
    image_model: str,
    video_model: str,
    resolution: str,
    duration: int,
    seed: int,
    endpoint: str,
    poll_interval: int,
    timeout_seconds: int,
) -> EatingVideoResult:
    api_key = read_api_key(key_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    keyframe_path = generate_eating_keyframe(
        pose_image_path=pose_image_path,
        api_key=api_key,
        output_dir=output_dir,
        model=image_model,
        seed=seed,
        endpoint=endpoint,
    )
    raw_video_path, video_response = submit_and_wait_video(
        keyframe_path=keyframe_path,
        api_key=api_key,
        output_dir=output_dir,
        model=video_model,
        resolution=resolution,
        duration=duration,
        seed=seed + 100,
        endpoint=endpoint,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
    )
    final_video_path = postprocess_video(
        raw_video_path,
        output_dir / "final" / "eat_5s_30fps_16x9.mp4",
    )

    metadata_path = output_dir / "metadata" / "eat_video_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pose_image": str(pose_image_path),
        "keyframe": str(keyframe_path),
        "raw_video": str(raw_video_path),
        "final_video": str(final_video_path),
        "image_model": image_model,
        "video_model": video_model,
        "resolution": resolution,
        "duration": duration,
        "seed": seed,
        "endpoint": endpoint,
        "video_task_id": get_video_response_field(video_response, "task_id"),
        "video_task_status": get_video_response_field(video_response, "task_status"),
        "usage": get_video_response_usage(video_response),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return EatingVideoResult(keyframe_path, raw_video_path, final_video_path, metadata_path)


def generate_running_video(
    *,
    pose_image_path: Path,
    key_file: Path,
    output_dir: Path,
    image_model: str,
    video_model: str,
    resolution: str,
    duration: int,
    seed: int,
    endpoint: str,
    poll_interval: int,
    timeout_seconds: int,
) -> RunningVideoResult:
    api_key = read_api_key(key_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    keyframe_path = generate_running_keyframe(
        pose_image_path=pose_image_path,
        api_key=api_key,
        output_dir=output_dir,
        model=image_model,
        seed=seed,
        endpoint=endpoint,
    )
    raw_video_path, video_response = submit_and_wait_video(
        keyframe_path=keyframe_path,
        api_key=api_key,
        output_dir=output_dir,
        model=video_model,
        resolution=resolution,
        duration=duration,
        seed=seed + 200,
        endpoint=endpoint,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
        prompt=RUNNING_VIDEO_PROMPT,
        negative_prompt=RUNNING_VIDEO_NEGATIVE_PROMPT,
        action_name="run",
    )
    final_video_path = postprocess_video(
        raw_video_path,
        output_dir / "final" / f"run_{duration}s_30fps_16x9.mp4",
    )

    metadata_path = output_dir / "metadata" / "run_video_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pose_image": str(pose_image_path),
        "keyframe": str(keyframe_path),
        "raw_video": str(raw_video_path),
        "final_video": str(final_video_path),
        "image_model": image_model,
        "video_model": video_model,
        "resolution": resolution,
        "duration": duration,
        "seed": seed,
        "endpoint": endpoint,
        "video_task_id": get_video_response_field(video_response, "task_id"),
        "video_task_status": get_video_response_field(video_response, "task_status"),
        "usage": get_video_response_usage(video_response),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return RunningVideoResult(keyframe_path, raw_video_path, final_video_path, metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an eating cat video from pose2 using DashScope Qwen + Wan.")
    parser.add_argument("--pose-image", type=Path, default=Path("outputs/cat_poses_comfy/transparent/pose_02_transparent.png"))
    parser.add_argument("--key-file", type=Path, default=Path("apikey.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eat_video_pose2"))
    parser.add_argument("--image-model", default="qwen-image-edit")
    parser.add_argument("--video-model", default="wan2.2-i2v-flash")
    parser.add_argument("--resolution", default="720P")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--endpoint", default="https://dashscope.aliyuncs.com/api/v1")
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_eating_video(
        pose_image_path=args.pose_image.resolve(),
        key_file=args.key_file.resolve(),
        output_dir=args.output_dir.resolve(),
        image_model=args.image_model,
        video_model=args.video_model,
        resolution=args.resolution,
        duration=args.duration,
        seed=args.seed,
        endpoint=args.endpoint,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"keyframe={result.keyframe_path}")
    print(f"raw_video={result.raw_video_path}")
    print(f"final_video={result.final_video_path}")
    print(f"metadata={result.metadata_path}")


if __name__ == "__main__":
    main()
