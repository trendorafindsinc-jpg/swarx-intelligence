# SWARX EMBER-0.1 Physical Offline E2E Verification

**Milestone:** EMBER Resilience Gate
**Status:** VERIFIED
**Verification type:** Physical network transition
**Model:** SWARX-EMBER-0.1

## Result

SWARX EMBER-0.1 successfully demonstrated real online-to-offline resilience using the actual SWARX HTTP transport and the local EMBER runtime.

### Online phase — PASS

The test executable reached the configured external HTTP endpoint through the real system `curl` transport.

Observed result:

```text
SWARX EMBER physical E2E ONLINE: PASS
Cloud path reached through real HTTP transport.
```

### Offline phase — PASS

Network connectivity was physically disabled on the development device. No `offline=true` simulation flag was used.

The real HTTP transport failed with DNS/network failure:

```text
curl: (6) Could not resolve host: example.com
```

The resilience gateway detected the cloud failure and activated the local EMBER runtime:

```text
SWARX EMBER physical E2E OFFLINE: PASS
Real network failure triggered local EMBER fallback.
```

An EMBER-generated response was returned after the cloud path became unavailable.

## Verified path

```text
SWARX
  -> Resilience Gateway
      -> Real HTTP/Cloud path
          -> network available: SUCCESS

      -> network failure
          -> SWARX EMBER-0.1
              -> SWARX Intelligence runtime
                  -> local model generation: SUCCESS
```

## What this milestone proves

- The real HTTP transport can reach an external endpoint while connectivity is available.
- A real network/DNS failure is observable by the SWARX runtime.
- The Resilience Gateway can transition from the failed cloud path to EMBER.
- SWARX-EMBER-0.1 can generate locally after the network path fails.
- The fallback does not require a hosted AI inference API.
- The resilience behavior is not dependent on a manually supplied offline simulation flag.

## Scope and limitation

This milestone verifies **resilience and local runtime activation**, not production intelligence quality.

SWARX-EMBER-0.1 is currently an exact-weight clone of the verified SWARX-0.1-Nano reference and has 541,312 parameters, 512 vocabulary entries, and 128-token context. Its response quality is therefore experimental and intentionally not treated as production-grade intelligence.

## Reproduction

Build:

```bash
cd ~/projects/swarx-intelligence
cmake -S . -B build
cmake --build build -j2
```

Online test with connectivity enabled:

```bash
./build/swarx-ember-physical-offline-e2e-tests online
```

Offline test with Wi-Fi and mobile data physically disabled:

```bash
./build/swarx-ember-physical-offline-e2e-tests offline
```

The offline test must be performed by actually removing network connectivity. Do not replace the physical transition with an application-level offline flag when using this verification as evidence of the resilience gate.

## Engineering decision

**SWARX EMBER-0.1 Physical Resilience Gate: PASS.**

This verification is frozen as an engineering milestone. Future changes to the resilience boundary, EMBER artifact, tokenizer compatibility, or runtime failover behavior must preserve or explicitly supersede this verification with a new recorded result.

> When the cloud goes silent, the Ember remains.
