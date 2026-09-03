#!/usr/bin/env python3
"""Measure a repeatable CPU inference baseline for the SWARX-0.1-Nano package."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT.parent / "swarx-ai-models"


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def measure(cli: Path, model: Path, vocab: Path, prompt: str, max_new: int, repeats: int) -> dict:
    samples: list[float] = []
    outputs: list[str] = []
    for _ in range(repeats):
        start = time.perf_counter()
        proc = subprocess.run(
            [str(cli), "--generate", prompt, "--max-new", str(max_new), "--model", str(model), "--vocab", str(vocab)],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "generation benchmark failed")
        line = next((x for x in proc.stdout.splitlines() if x.startswith("GENERATED:")), None)
        if line is None:
            raise RuntimeError("benchmark output missing GENERATED line")
        outputs.append(line[len("GENERATED: "):])
        samples.append(elapsed)

    return {
        "prompt": prompt,
        "max_new_tokens": max_new,
        "repeats": repeats,
        "seconds": samples,
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
        "p95_seconds": percentile(samples, 0.95),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "output": outputs[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, default=ROOT / "build" / "swarx-cli")
    parser.add_argument("--package", type=Path, default=DEFAULT_MODELS / "exports" / "swarx-0.1-nano")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "swarx-0.1-nano-performance.json")
    args = parser.parse_args()

    model = args.package / "model.swarx"
    vocab = args.package / "tokenizer.vocab"
    if not args.cli.exists() or not model.exists() or not vocab.exists():
        raise RuntimeError("missing CLI or SWARX package; build the engine and provide a valid package")
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")

    prompts = ["hello", "What is a budget?", "I am SWARX"]
    results = [measure(args.cli, model, vocab, p, args.max_new, args.repeats) for p in prompts]
    report = {
        "title": "SWARX-0.1-Nano PERFORMANCE BASELINE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "SWARX Intelligence native CPU CLI",
        "platform": "runtime host",
        "cli": str(args.cli),
        "package": str(args.package),
        "measurement": "wall-clock process-to-process generation latency including model load",
        "note": "This is a baseline, not a hardware-normalized benchmark. Run on the target device and retain the raw samples.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("SWARX-0.1-Nano PERFORMANCE BASELINE: PASS")
    for result in results:
        print(
            f"{result['prompt']!r}: mean={result['mean_seconds']:.4f}s "
            f"median={result['median_seconds']:.4f}s p95={result['p95_seconds']:.4f}s"
        )
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SWARX-0.1-Nano PERFORMANCE BASELINE: FAIL — {exc}")
        raise SystemExit(1)
