# SWARX Intelligence

SWARX Intelligence is the native intelligence and inference-engine foundation for SWARX.

## 🎉 Milestone achieved — SWARX-0.1-Nano VERIFIED

**September 2026 — Foundation integration milestone complete.**

SWARX Intelligence has successfully completed the consolidated verification of the first SWARX-owned model pipeline: **SWARX-0.1-Nano — 6/6 gates passed.**

This is the first verified proof that a SWARX-owned trained model can be exported from the model repository, loaded by the native intelligence engine, executed, and verified across Python and C++.

### Verification record

| Gate | Result |
|---|---:|
| Native CTest suite | **13/13 PASS** |
| End-to-end engine integration | **PASS** |
| Python/C++ tokenizer compatibility | **16/16 PASS** |
| Python/C++ logits compatibility | **6/6 PASS** |
| Python/C++ generation compatibility | **6/6 PASS** |
| Performance baseline | **PASS** |
| **Consolidated verification** | **6/6 PASS** |

### Verified model baseline

- Model: **SWARX-0.1-Nano**
- Architecture: decoder-only Transformer
- Parameters: **541,312**
- Vocabulary: **512 tokens**
- Hidden size: **128**
- Intermediate size: **512**
- Layers: **2**
- Attention heads: **4**
- Context length: **128**
- Export format: **SWARX v2**
- Tensor schema: **v2**
- Tokenizer: SentencePiece BPE
- Locked tokenizer SHA256: `73f38bddc6f329aa1e2e442882644e516fc4923b3b02e3004258970326f75307`
- Verified artifact SHA256: `4c9d754d095c2f20217f692c2361280dc14e72cb874efb155f82dd71cd22cbd6`

### Performance baseline

Process-to-process measurements, including model loading:

- `hello`: mean **0.3815s**, p95 **0.3914s**
- `budget`: mean **0.4092s**, p95 **0.4170s**
- `I am SWARX`: mean **0.1258s**, p95 **0.1358s**

### What the milestone proves

```text
swarx-ai-models
      │
      │  train / checkpoint / export
      ▼
SWARX-0.1-Nano
      │
      │  SWARX v2 artifact
      ▼
SWARX Intelligence
      │
      ├── tokenize
      ├── load
      ├── inference
      └── generate
      │
      ▼
  verification
      │
      └── Python ↔ C++ compatibility
```

The Nano model is deliberately tiny and is **not a production-scale LLM**. Its purpose is to establish a trustworthy native execution and compatibility baseline before the model is scaled substantially.

---

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
- **Structured intelligence bus for model/tool communication and provenance**
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

## Intelligence bus

Models and tools communicate through structured envelopes rather than hidden free-form conversations. A request can produce observations, tool results, model responses, verification messages and a final response while retaining request IDs and source provenance.

```text
REQUEST
   |
   +--> TOOL / KNOWLEDGE OBSERVATION
   |
   +--> MODEL RESPONSE
   |
   +--> VERIFICATION
   |
   +--> FINAL RESPONSE

Every message carries:
request_id / sender / task / payload / capabilities / sources / confidence
```

This is the foundation for future multi-model orchestration without forcing one model provider to own the whole system.

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
│   ├── tool_registry
│   └── intelligence_bus
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

**SWARX-0.1-Nano integration: VERIFIED — 6/6 consolidated gates passed.**

This repository is an engineering foundation, not yet a production-scale trained LLM. The native runtime and intelligence architecture now have a verified SWARX-owned model running end-to-end. The next stage is to preserve this baseline and scale toward a substantially more capable SWARX model.

The verified loop is:

```text
MODEL
  -> LOAD
  -> TOKENIZE
  -> INFERENCE
  -> GENERATE
  -> STREAM
  -> ANSWER
```

SWARX can now build on this verified foundation for larger models, stronger reasoning, memory, tools, routing, optimized kernels/backends and product integrations.

## SWARX ownership direction

The repository is designed so SWARX can own the runtime, model format, execution APIs, routing, online-source policy, optimization layers and product integrations independently. Third-party models and providers may be adapted later, but they do not define the core architecture.
