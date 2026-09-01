# SWARX Intelligence

SWARX Intelligence is the native inference-engine project for SWARX.

## Goal

Build a modular, hardware-aware AI runtime owned by SWARX rather than coupling the core architecture to a third-party inference engine.

## Current foundation

- C++20 native core
- Tensor abstraction and tensor math
- Deterministic tokenizer API boundary
- Greedy, temperature, top-k/top-p and repetition-aware sampling
- Transformer execution and causal attention
- Model loading/serialization and model inspection
- Model identity/fingerprinting
- KV-cache management and sliding-context generation
- Inference sessions, streaming, cancellation and batch generation
- Role-aware conversational chat runtime
- Stable product-facing runtime facade
- **New: pluggable compute-backend abstraction with native CPU backend**
- **New: symmetric INT8 quantization/dequantization with error statistics**
- CMake build and automated CTest coverage
- CLI smoke application

## Architecture

```text
SWARX Intelligence
├── core/       tensors, math, tokenizer, sampling
├── runtime/    model execution, generation, chat, scheduling
├── formats/    model loading and serialization
├── backends/   CPU / mobile / optional accelerator backends
├── models/     model adapters, identity and metadata
├── quant/      compact INT8 representation and calibration metrics
└── api/        stable engine interface for SWARX products
```

The runtime is intentionally owned and modular. Compute operations now have a backend seam, while the first implementation is a portable CPU backend. INT8 quantization is available as an independent primitive so future model loaders and optimized backends can adopt compact weights without redesigning the public model API.

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

The project is an **engineering foundation, not yet a production-scale trained LLM**. The runtime architecture is now substantially closer to a sellable/embeddable native engine, but model quality still depends on trained model weights and future optimized kernels/backends.

## SWARX ownership direction

The repository is designed so SWARX can own the runtime, model format, execution APIs, optimization layers and product integrations independently. Third-party models may be adapted later, but the core execution architecture does not require a third-party inference engine.
