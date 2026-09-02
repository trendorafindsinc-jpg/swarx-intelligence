#!/usr/bin/env python3
"""Compare Python nano logits with the native SWARX Intelligence runtime."""
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
    module_name = "swarx_nano_pipeline"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load nano pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses (Python 3.14+) expects the module to be registered while
    # executing a dynamically loaded module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def python_logits(models_repo: Path, checkpoint: Path, text: str, vocab_path: Path):
    nano = load_nano_module(models_repo)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = nano.NanoConfig(**payload["config"])
    model = nano.NanoTransformer(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    import sentencepiece as spm

    model_file = vocab_path.with_suffix(".model")
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    ids = sp.encode(text, out_type=int, add_bos=True, add_eos=False)
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(input_ids)
    return ids, logits[0, -1].tolist()


def cpp_logits(cli: Path, model: Path, vocab: Path, text: str):
    proc = subprocess.run(
        [str(cli), "--logits", text, "--model", str(model), "--vocab", str(vocab)],
        check=True,
        capture_output=True,
        text=True,
    )
    line = next((line for line in proc.stdout.splitlines() if line.startswith("LOGITS:")), None)
    if line is None:
        raise RuntimeError(f"C++ logits output missing. stdout={proc.stdout!r} stderr={proc.stderr!r}")
    values = [float(x) for x in line.split()[1:]]
    return values


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
    args = parser.parse_args()

    cases = [
        "hello",
        "I am SWARX",
        "What is a budget?",
        "2 + 2 = 4",
        "Nigeria naira ₦",
        "café résumé",
    ]

    print("SWARX Python/C++ logits compatibility gate")
    print(f"Intelligence repo: {ROOT}")
    print(f"Models repo:       {args.models_repo}")
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"SWARX artifact:    {args.model}")
    print(f"Cases:             {len(cases)}")

    max_abs = 0.0
    max_mean = 0.0
    for text in cases:
        py_ids, py = python_logits(args.models_repo, args.checkpoint, text, args.vocab)
        cpp = cpp_logits(args.cli, args.model, args.vocab, text)
        if len(py) != len(cpp):
            raise AssertionError(f"{text!r}: shape mismatch Python={len(py)} C++={len(cpp)}")

        diffs = [abs(a - b) for a, b in zip(py, cpp)]
        case_max = max(diffs)
        case_mean = sum(diffs) / len(diffs)
        max_abs = max(max_abs, case_max)
        max_mean = max(max_mean, case_mean)

        ok = case_max <= 1e-4 and case_mean <= 1e-5
        print(f"{'PASS' if ok else 'FAIL'}: {text!r} tokens={py_ids} max_abs={case_max:.8g} mean_abs={case_mean:.8g}")
        if not ok:
            return 1

    print(f"Max absolute difference: {max_abs:.8g}")
    print(f"Max mean absolute diff:  {max_mean:.8g}")
    print("RESULT: PASS — Python and C++ logits match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
