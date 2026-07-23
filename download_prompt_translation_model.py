from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from prompt_translator import translate


REPOSITORY = "Helsinki-NLP/opus-mt-zh-en"
REVISION = "cf109095479db38d6df799875e34039d4938aaa6"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the pinned offline Chinese-to-English translator.",
    )
    parser.add_argument("--target", type=Path, required=True)
    arguments = parser.parse_args()
    target = arguments.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    ready_marker = target / "translation_ready.json"
    ready_marker.unlink(missing_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=target,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "source.spm",
            "target.spm",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )
    required = [
        target / "config.json",
        target / "source.spm",
        target / "target.spm",
        target / "vocab.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    weights = [target / "pytorch_model.bin", target / "model.safetensors"]
    if not any(path.is_file() for path in weights):
        missing.append(str(target / "{pytorch_model.bin|model.safetensors}"))
    if missing:
        raise FileNotFoundError("Downloaded translator is incomplete: " + ", ".join(missing))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    verification_source = "猫咪正在吃饭"
    verification = translate(target, [verification_source])[0].strip()
    if not verification or any("\u3400" <= char <= "\u9fff" for char in verification):
        raise RuntimeError("Translator verification did not produce valid English text.")
    ready_marker.write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "verified_source": verification_source,
                "verified_translation": verification,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
