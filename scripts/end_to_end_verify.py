#!/usr/bin/env python3
"""End-to-end SWARX-0.1-Nano model-to-engine verification.

The gate intentionally requires the packaged artifact to match the locked model
contract. It does not modify or regenerate the model or tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT.parent / "swarx-ai-models"
EXPECTED_TOKENIZER_SHA256 = "73f38bddc6f329aa1e2e442882644e516fc4923b3b02e3004258970326f75307"


def run(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {label} ===")
    print("$", " ".join(command))
    return subprocess.run(command, text=True, capture_output=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-repo", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--cli", type=Path, default=ROOT / "build" / "swarx-cli")
    parser.add_argument("--package", type=Path, default=DEFAULT_MODELS / "exports" / "swarx-0.1-nano")
    parser.add_argument("--max-new", type=int, default=24)
    args = parser.parse_args()

    package = args.package
    model = package / "model.swarx"
    vocab = package / "tokenizer.vocab"
    tokenizer_model = package / "tokenizer" / "swarx-0.1-nano.model"
    contract = args.models_repo / "configs" / "swarx-0.1-nano.contract.json"
    validator = args.models_repo / "export" / "validate_export.py"

    for path, label in [
        (args.cli, "SWARX CLI"),
        (model, "SWARX artifact"),
        (vocab, "packaged tokenizer vocab"),
        (tokenizer_model, "packaged tokenizer model"),
        (contract, "model contract"),
        (validator, "export validator"),
    ]:
        require_file(path, label)

    print("SWARX-0.1-Nano ENGINE INTEGRATION")
    print(f"Models repo: {args.models_repo}")
    print(f"Intelligence repo: {ROOT}")
    print(f"Package: {package}")

    locked_tok = EXPECTED_TOKENIZER_SHA256
    actual_tok = sha256(tokenizer_model)
    print(f"Tokenizer SHA-256: {actual_tok}")
    if actual_tok != locked_tok:
        raise RuntimeError(
            "STALE FIXTURE: packaged tokenizer does not match the locked "
            f"SWARX-0.1-Nano identity: actual={actual_tok} expected={locked_tok}"
        )

    validation = run(
        [
            sys.executable,
            str(validator),
            str(package),
            "--contract",
            str(contract),
        ],
        "Package + contract validation",
    )
    if validation.returncode != 0:
        sys.stderr.write(validation.stdout)
        sys.stderr.write(validation.stderr)
        raise RuntimeError("package contract validation failed")
    print(validation.stdout.strip())

    prompt = "What is a budget?"
    tokenize = run(
        [str(args.cli), "--tokenize", prompt, "--vocab", str(vocab)],
        "Tokenizer -> token IDs",
    )
    if tokenize.returncode != 0:
        raise RuntimeError(tokenize.stderr.strip() or "tokenization failed")
    token_line = next((x for x in tokenize.stdout.splitlines() if x.startswith("TOKENS:")), None)
    if not token_line:
        raise RuntimeError("tokenizer output missing TOKENS line")
    ids = [int(x) for x in token_line.split()[1:]]
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    vocab_size = int(manifest["model"]["vocab_size"])
    if not ids or any(i < 0 or i >= vocab_size for i in ids):
        raise RuntimeError(f"token IDs out of range: {ids}")
    print(f"PASS: {len(ids)} token IDs in [0, {vocab_size})")

    logits = run(
        [str(args.cli), "--logits", prompt, "--model", str(model), "--vocab", str(vocab)],
        "Model load -> forward -> logits",
    )
    if logits.returncode != 0:
        raise RuntimeError(logits.stderr.strip() or "logits execution failed")
    logits_line = next((x for x in logits.stdout.splitlines() if x.startswith("LOGITS:")), None)
    if not logits_line:
        raise RuntimeError("logits output missing LOGITS line")
    values = [float(x) for x in logits_line.split()[1:]]
    if len(values) != vocab_size:
        raise RuntimeError(f"logit count {len(values)} != vocab size {vocab_size}")
    if not all(__import__("math").isfinite(x) for x in values):
        raise RuntimeError("non-finite logit detected")
    print(f"PASS: {len(values)} finite logits")

    generation = run(
        [
            str(args.cli),
            "--generate",
            prompt,
            "--max-new",
            str(args.max_new),
            "--model",
            str(model),
            "--vocab",
            str(vocab),
        ],
        "Generation -> decode",
    )
    if generation.returncode != 0:
        raise RuntimeError(generation.stderr.strip() or "generation failed")
    generated = next((x[len("GENERATED: ") :] for x in generation.stdout.splitlines() if x.startswith("GENERATED:")), None)
    if generated is None:
        raise RuntimeError("generation output missing GENERATED line")
    if not generated.strip():
        raise RuntimeError("generation returned empty text")
    print(f"PASS: generated text: {generated!r}")

    print("\nSWARX-0.1-Nano ENGINE INTEGRATION: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SWARX-0.1-Nano ENGINE INTEGRATION: FAIL — {exc}", file=sys.stderr)
        raise SystemExit(1)
