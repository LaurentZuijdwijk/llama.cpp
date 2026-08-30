# Merge origin/master (0190529ec) into vulkan/qwen4exp-rocmfpx

Branch: merge/upstream-0190529ec   Worktree: ~/upstream-merge
Merge base: d222767c7 (Aug 25).  92 local ahead / 76 upstream behind.

## Root cause of the conflicts
Three PRs this fork carried early have now landed upstream in final form:
  #27742 qwen4exp   -> local 651a5b191..b98aa9847
  #27342 DFlash2    -> local 5ecbe1ac1
  #27795/#27794/#27762 quantize cap, TENSOR_READ_LAZY, KV token IDs

## Resolved (15/19)
theirs, no loss:
  src/models/dflash.cpp        both local fixes already upstream, byte-identical
  include/llama.h              adopts upstream --lazy-mode rename
  src/llama-model.cpp          "
  src/llama-model-loader.{cpp,h}  upstream lazy_read struct supersedes lazy_tensor_ranges
  conversion/qwen.py           upstream-only addition
  tests/test-llama-archs.cpp   upstream owns DEEPSEEK4 now
  gguf-py/gguf/gguf_writer.py  Keys.PLE -> Keys.PerLayerEmbedding (symbol only)

theirs + local patch re-applied:
  common/speculative.cpp       upstream base + ca2616991 (--spec-draft-adaptive),
                               ctor now takes upstream's new n_max member
  src/llama-quant.cpp          upstream row-slab streaming (#27795) supersedes local
                               row bands and also caps the source read; kept the
                               ROCmFPx requantize exemption; dropped stale stream_out

ours:
  src/llama-kv-cells.h         for_each_token_in_reverse, pure addition
  src/llama-kv-cache.cpp       reverse-scan get_prev_tokens on top of upstream

union:
  tools/server/server-context.cpp  upstream synth-probs path + local spec_is_replay
                                   fix + spec_dists path
  tests/test-backend-ops.cpp       both top_k case sets
  gguf-py/gguf/constants.py        upstream PER_LAYER_TOKEN_EMBD + local
                                   PLE_NGRAM_EMBD / NEXTN_*

## Remaining (4) - the qwen4exp core, needs design decisions
  src/models/qwen4exp.cpp          27 hunks / 805 lines  (add/add, no common base)
  src/llama-memory-hybrid-idx.cpp  18 hunks / 337 lines  (add/add)
  src/llama-memory-hybrid-idx.h     4 hunks /  73 lines  (add/add)
  src/models/models.h               5 hunks /  46 lines

Local qwen4exp extensions to re-express on upstream's structure:
  MTP draft head, PLE-on-disk (--model-ple/--ngram-on-disk), QSA pooled-key cache,
  n-gram history from the KV cache, non-unified indexer cache, per-head PLE split.

## Verified compat
GGUF KV key strings ("{arch}.ple.*") unchanged by the upstream rename - existing
converted files still load.
