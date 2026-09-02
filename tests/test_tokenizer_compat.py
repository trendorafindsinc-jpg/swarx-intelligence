"""Verify Python SWARX tokenizer IDs match the native C++ tokenizer.

This cross-engine gate belongs in swarx-intelligence because it verifies the
contract consumed by the native runtime. The Python tokenizer implementation
is loaded directly from swarx-ai-models/model/tokenizer.py so this test does
not import the training package or require unrelated dependencies such as
PyYAML.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tokenizer_class(tokenizer_py: Path):
    spec = importlib.util.spec_from_file_location("swarx_model_tokenizer", tokenizer_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tokenizer module: {tokenizer_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SwarxTokenizer


CASES = [
    "",
    "hello",
    "Hello",
    "Hello SWARX",
    "I am SWARX",
    "What is a budget?",
    "User: Who are you?\nAssistant:",
    "save 100 and spend 40",
    "2 + 2 = 4",
    "5 / 4",
    "Nigeria naira ₦",
    "café résumé",
    "こんにちは",
    "emoji 😀",
    "tabs\tand spaces",
    "multiple   spaces",
]


def run_cpp(cli: Path, vocab: Path, text: str) -> list[int]:
    completed = subprocess.run(
        [str(cli), "--tokenize", text, "--vocab", str(vocab)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    token_lines = [line for line in lines if line.startswith("TOKENS:")]
    if len(token_lines) != 1:
        raise RuntimeError(f"Unexpected C++ tokenizer output: {completed.stdout!r}")
    payload = token_lines[0][len("TOKENS:") :].strip()
    return [int(value) for value in payload.split()] if payload else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True, help="native swarx-cli executable")
    parser.add_argument(
        "--models-repo",
        type=Path,
        default=ROOT.parent / "swarx-ai-models",
        help="local swarx-ai-models checkout",
    )
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    args = parser.parse_args()

    models_repo = args.models_repo.expanduser().resolve()
    tokenizer_py = models_repo / "model" / "tokenizer.py"
    tokenizer_model = args.tokenizer or (models_repo / "tokenizer" / "swarx-0.1-nano.model")
    vocab = args.vocab or (models_repo / "tokenizer" / "swarx-0.1-nano.vocab")

    if not tokenizer_py.exists():
        raise FileNotFoundError(f"Python tokenizer source not found: {tokenizer_py}")
    if not tokenizer_model.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {tokenizer_model}")
    if not vocab.exists():
        raise FileNotFoundError(f"C++ vocab not found: {vocab}")

    SwarxTokenizer = load_tokenizer_class(tokenizer_py)
    tokenizer = SwarxTokenizer.load(tokenizer_model)
    failures: list[tuple[str, list[int], list[int]]] = []

    print("SWARX tokenizer compatibility gate")
    print(f"Intelligence repo: {ROOT}")
    print(f"Models repo:       {models_repo}")
    print(f"Python tokenizer:  {tokenizer_model}")
    print(f"C++ vocab:         {vocab}")
    print(f"Cases:             {len(CASES)}")
    print()

    for text in CASES:
        py_ids = tokenizer.encode(text, add_bos=True, add_eos=False)
        cpp_ids = run_cpp(args.cli, vocab, text)
        status = "PASS" if py_ids == cpp_ids else "FAIL"
        print(f"{status}: {text!r}")
        if status == "FAIL":
            print(f"  Python: {py_ids}")
            print(f"  C++:    {cpp_ids}")
            failures.append((text, py_ids, cpp_ids))

    print()
    print(f"Passed: {len(CASES) - len(failures)}/{len(CASES)}")
    if failures:
        print("RESULT: FAIL — Python and C++ token IDs diverge.")
        return 1

    print("RESULT: PASS — Python and C++ token IDs are identical for all cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
