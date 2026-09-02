#!/usr/bin/env python3
"""Compare deterministic Python nano generation with native SWARX generation."""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT.parent / "swarx-ai-models"


def load_nano_module(models_repo: Path):
    path = models_repo / "scripts" / "run_nano_pipeline.py"
    module_name = "swarx_nano_pipeline_generation"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load nano pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def python_generate(models_repo: Path, checkpoint: Path, text: str, max_new: int, tokenizer_model: Path):
    nano = load_nano_module(models_repo)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = nano.NanoConfig(**payload["config"])
    model = nano.NanoTransformer(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    return nano.generate(model, sp, text, max_new=max_new, device="cpu")


def cpp_generate(cli: Path, model: Path, vocab: Path, text: str, max_new: int):
    proc = subprocess.run(
        [
            str(cli),
            "--generate",
            text,
            "--max-new",
            str(max_new),
            "--model",
            str(model),
            "--vocab",
            str(vocab),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = next((line for line in proc.stdout.splitlines() if line.startswith("GENERATED:")), None)
    if line is None:
        raise RuntimeError(f"C++ generation output missing. stdout={proc.stdout!r} stderr={proc.stderr!r}")
    return line[len("GENERATED: "):]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=ROOT / "build" / "swarx-cli")
    parser.add_argument("--models-repo", type=Path, default=DEFAULT_MODELS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_MODELS / "checkpoints" / "swarx-0.1-nano" / "final.pt",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODELS / "exports" / "swarx-0.1-nano" / "model.swarx",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_MODELS / "exports" / "swarx-0.1-nano" / "tokenizer.vocab",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=DEFAULT_MODELS / "exports" / "swarx-0.1-nano" / "tokenizer" / "swarx-0.1-nano.model",
    )
    parser.add_argument("--max-new", type=int, default=40)
    args = parser.parse_args()

    cases = [
        "hello",
        "I am SWARX",
        "What is a budget?",
        "2 + 2 = 4",
        "Nigeria naira ₦",
        "café résumé",
    ]

    print("SWARX Python/C++ generation compatibility gate")
    print(f"Intelligence repo: {ROOT}")
    print(f"Models repo:       {args.models_repo}")
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"SWARX artifact:    {args.model}")
    print(f"Max new tokens:    {args.max_new}")
    print(f"Cases:             {len(cases)}")

    for text in cases:
        py = python_generate(args.models_repo, args.checkpoint, text, args.max_new, args.tokenizer_model)
        cpp = cpp_generate(args.cli, args.model, args.vocab, text, args.max_new)
        ok = py == cpp
        print(f"{'PASS' if ok else 'FAIL'}: {text!r}")
        if not ok:
            print(f"  Python: {py!r}")
            print(f"  C++:    {cpp!r}")
            return 1
        print(f"  Output: {py!r}")

    print("RESULT: PASS — Python and C++ deterministic generation match exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
