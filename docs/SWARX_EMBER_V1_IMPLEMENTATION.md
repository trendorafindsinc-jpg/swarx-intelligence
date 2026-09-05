# SWARX EMBER v1 — Implementation Milestone

## Purpose

This document records the first implementation step toward SWARX EMBER v1, the persistent local intelligence path defined by the canonical EMBER architecture specification in `swarx-ai-models`.

## Implemented

### Resilience Gateway contract

`include/swarx/resilience_gateway.hpp` and `src/runtime/resilience_gateway.cpp` add a SWARX-owned resilience boundary with:

- replaceable cloud adapter interface
- explicit cloud/EMBER/unavailable paths
- typed failure classes
- configurable cloud and EMBER enablement
- network policy enforcement
- optional single cloud retry
- local EMBER fallback after cloud failure
- local model health/integrity checks before fallback
- explicit indication when EMBER is activated
- honest degradation for fresh-information requests

### Integration

The gateway uses the existing `Runtime` facade for local inference. It does not introduce a second inference engine and does not replace the existing Transformer runtime.

The gateway is intentionally provider-neutral: cloud provider and model selection remain outside this layer and can be supplied through the injected adapter by the product-facing application.

### Test coverage

`tests/resilience_gateway_smoke.cpp` covers:

- successful cloud routing
- complete no-path behavior when cloud is disabled and EMBER is not loaded
- cloud retry behavior for a provider failure
- failure preservation when no local fallback is available

## Next engineering gates

1. Load the verified SWARX-0.1-Nano artifact through `ResilienceGateway` as the current local engineering reference.
2. Add an EMBER health/handshake contract with runtime, model, integrity, memory and network-required fields.
3. Add session-level local memory/persistence without modifying model weights.
4. Connect the product gateway so cloud provider/model settings are environment-driven.
5. Add a real network-disabled end-to-end test: online -> offline -> EMBER -> online recovery.
6. Add local/offline benchmark metrics: startup, first token, tokens/sec, peak RAM and model artifact size.
7. Freeze the v1 resilience contract before beginning EMBER v2 adaptive-learning work.

## Safety boundary

EMBER v1 does **not** autonomously modify model weights. Personalization is limited to controlled local state and memory until a separately specified and evaluated adaptive-learning system is approved for a future EMBER version.
