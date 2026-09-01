# SWARX-0.1 Compatibility Audit

The current runtime already implements the model-side tensor contract used by SWARX-0.1:

- decoder-only Transformer
- pre-RMSNorm blocks
- GELU feed-forward
- separate Q/K/V/output matrices
- explicit `[vocab, hidden]` language-model head
- SWARX weights format version 2

The default `ModelConfig` dimensions are now aligned to SWARX-0.1: vocabulary 8192,
hidden 512, intermediate 2048, 8 heads, 8 layers, context 2048.

## Tokenizer blocker

The current runtime tokenizer remains a bootstrap whitespace tokenizer. SWARX-0.1 training
currently produces a SentencePiece artifact. These are not equivalent tokenization systems,
so cross-engine token-ID and logit equivalence cannot honestly be claimed yet.

This branch deliberately records the mismatch instead of silently pretending compatibility.
