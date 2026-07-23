from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def translate(model_path: Path, texts: list[str]) -> list[str]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(max(1, min(16, int(os.environ.get("OMP_NUM_THREADS", "4")))))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            num_beams=4,
            do_sample=False,
            max_length=512,
            early_stopping=True,
        )
    return [
        text.strip()
        for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline Chinese-to-English prompt translator.",
    )
    parser.add_argument("--model", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        payload: Any = json.load(sys.stdin)
        texts = payload["texts"]
        if (
            not isinstance(texts, list)
            or not 1 <= len(texts) <= 16
            or not all(isinstance(text, str) and text.strip() for text in texts)
        ):
            raise ValueError("texts must contain 1 to 16 non-empty strings")
        translations = translate(arguments.model.resolve(), texts)
        if len(translations) != len(texts) or any(not text for text in translations):
            raise RuntimeError("translation output is incomplete")
        json.dump(
            {"translations": translations},
            sys.stdout,
            ensure_ascii=False,
        )
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must return a compact failure
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
