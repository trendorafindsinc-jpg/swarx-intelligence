# SWARX Intelligence

SWARX Intelligence is the native inference-engine project for SWARX.

## Goal

Build a small, modular, hardware-aware AI runtime owned by SWARX rather than coupling the core architecture to a third-party inference engine.

## Current foundation

- C++20 core
- Minimal tensor abstraction
- Deterministic tokenizer API boundary
- Greedy and temperature sampling
- Engine/runtime API
- CLI smoke application
- CMake build and CTest smoke test

## Architecture direction

```text
SWARX Intelligence
├── core/       math, tensors, tokenizer, sampling
├── runtime/    model execution and scheduling
├── formats/    model loading and serialization
├── backends/   CPU / mobile / optional accelerator backends
├── models/     model adapters and metadata
└── api/        stable engine interface for SWARX products
```

The project starts intentionally small. Model loading, transformer execution, quantization, KV-cache management, and hardware acceleration will be added behind stable interfaces instead of importing a large external engine into the repository.

## Build

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## Run

```bash
./build/swarx-cli
```

## Status

Early foundation — the runtime shell is working, but this is **not yet a production LLM inference engine**. The next milestones are tensor kernels, model format loading, transformer blocks, KV cache, and CPU/mobile optimization.
