#!/usr/bin/env python3
"""Run the consolidated SWARX-0.1-Nano verification gates.

This command is intentionally strict: a stale or incompatible model package is
reported as NOT VERIFIED rather than being silently accepted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> bool:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print("$", " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    status = result.returncode == 0
    print(f"{label}: {'PASS' if status else 'NOT VERIFIED'}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-repo", type=Path, default=ROOT.parent / "swarx-ai-models")
    parser.add_argument("--cli", type=Path, default=ROOT / "build" / "swarx-cli")
    parser.add_argument("--package", type=Path, default=None)
    parser.add_argument("--skip-ctest", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    args = parser.parse_args()

    package = args.package or (args.models_repo / "exports" / "swarx-0.1-nano")
    results: list[bool] = []

    if not args.skip_ctest:
        results.append(run("Native CTest suite", ["ctest", "--test-dir", str(ROOT / "build"), "--output-on-failure"]))

    results.append(run(
        "End-to-end engine integration",
        [sys.executable, str(ROOT / "scripts" / "end_to_end_verify.py"),
         "--models-repo", str(args.models_repo), "--cli", str(args.cli), "--package", str(package)],
    ))

    results.append(run(
        "Tokenizer Python/C++ compatibility",
        [sys.executable, str(ROOT / "tests" / "test_tokenizer_compat.py"),
         "--cli", str(args.cli), "--models-repo", str(args.models_repo)],
    ))

    results.append(run(
        "Logits Python/C++ compatibility",
        [sys.executable, str(ROOT / "tests" / "test_logits_compat.py"),
         "--cli", str(args.cli), "--models-repo", str(args.models_repo)],
    ))

    results.append(run(
        "Generation Python/C++ compatibility",
        [sys.executable, str(ROOT / "tests" / "test_generation_compat.py"),
         "--cli", str(args.cli), "--models-repo", str(args.models_repo)],
    ))

    if not args.skip_performance:
        results.append(run(
            "Performance baseline",
            [sys.executable, str(ROOT / "scripts" / "performance_baseline.py"),
             "--cli", str(args.cli), "--package", str(package)],
        ))

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 72)
    print(f"SWARX-0.1-Nano CONSOLIDATED VERIFICATION: {'PASS' if passed == total else 'NOT VERIFIED'}")
    print(f"Gates: {passed}/{total} passed")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
