# SWARX Intelligence

SWARX Intelligence is the native intelligence and inference-engine foundation for SWARX.

## Mission

Build a modular, hardware-aware intelligence runtime owned by SWARX rather than coupling the core architecture to a single third-party model or AI API.

External models and online providers are optional adapters. They are replaceable infrastructure, not the foundation of SWARX.

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
- Pluggable compute-backend abstraction with native CPU backend
- Symmetric INT8 quantization/dequantization with error statistics
- Serving/runtime pipeline primitives
- Dependency-light HTTPS transport
- Controlled online information gateway
- **Policy-aware intelligence source router**
- **Central capability/tool registry**
- CMake build and automated CTest coverage
- CLI smoke application

## Online intelligence architecture

SWARX does not treat every question as a generic web-search request. The intelligence layer classifies the information requirement and selects an appropriate controlled source.

```text
                         SWARX
                           |
                  SWARX INTELLIGENCE
                           |
                  INTELLIGENCE ROUTER
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
   OWNED KNOWLEDGE    LIVE UTILITIES    EXTERNAL ADAPTERS
          |                |                |
          v          +-----+-----+     +----+----+
   TrendoraFinds     |           |     |         |
                     v           v     v         v
                  Time       Weather  Web      Finance/News
```

### TrendoraFinds

TrendoraFinds is the SWARX ecosystem's owned knowledge source for finance, education and related business content. It is not the same thing as the general internet and it does not replace SWARX Intelligence.

SWARX retrieves from TrendoraFinds when that source is appropriate, then the model/runtime can reason over the retrieved material.

### Live information

Volatile information should come from live sources instead of model memory. The current gateway has controlled routes for:

- Time — TimeAPI
- Weather/forecast — Open-Meteo
- TrendoraFinds — WordPress REST knowledge endpoint

Additional source adapters are represented in the capability registry for web search, finance data and news. Those are intentionally disabled until a production-grade provider and permission model are selected.

## Capability security direction

The model is not given unrestricted URL access.

```text
REQUEST
   |
IDENTIFY CAPABILITY
   |
POLICY CHECK
   |
SOURCE ROUTER
   |
CONTROLLED TOOL
   |
PROVENANCE
   |
MODEL / SYNTHESIS
```

Every online result carries source identity, URL, HTTP status and retrieval timestamp so product layers can eventually disclose provenance.

## Architecture

```text
SWARX Intelligence
├── core/
│   ├── tensors
│   ├── math
│   ├── tokenizer
│   └── sampling
├── runtime/
│   ├── transformer
│   ├── attention
│   ├── generation
│   ├── inference
│   ├── kv_cache
│   ├── chat
│   ├── serving
│   ├── online
│   ├── intelligence_router
│   └── tool_registry
├── models/
│   ├── model adapters
│   ├── identity
│   └── inspection
├── backends/
│   └── CPU / future mobile / accelerator backends
├── quantization/
│   └── compact INT8 primitives
└── product API/
    └── stable runtime facade
```

## Build

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Network-dependent tests are opt-in so ordinary development does not silently consume mobile data:

```bash
SWARX_RUN_ONLINE_TESTS=1 ctest --test-dir build --output-on-failure
```

## Status

This repository is an **engineering foundation, not yet a production-scale trained LLM**. The native runtime and intelligence architecture are becoming suitable for an owned, embeddable SWARX engine, but model quality still depends on trained weights and future optimized kernels/backends.

The immediate engineering objective remains:

```text
MODEL
  -> LOAD
  -> TOKENIZE
  -> INFERENCE
  -> GENERATE
  -> STREAM
  -> ANSWER
```

Once a real model runs end-to-end, the next layers can safely build on the model contract, routing, memory, tools, verification and local/cloud execution.

## SWARX ownership direction

The repository is designed so SWARX can own the runtime, model format, execution APIs, routing, online-source policy, optimization layers and product integrations independently. Third-party models and providers may be adapted later, but they do not define the core architecture.
