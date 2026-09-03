# Qwen3.8 Flash-Next (qwen4exp) performance session, 2026-09-02

Branch `qwen-flash-next-performance`, base `80acfa246`. Model: `agentionai/Qwen3.8-Flash-Next-AP-GGUF`
AP-IQ4_XS (experts IQ3_S/IQ4_NL, dense backbone Q6_K, router F32, ~61 GB on device, PLE table on
disk). Draft: `Qwen3.8-Flash-Next-MTP-Q8_0`. Box: Strix Halo / Radeon 8060S, Vulkan RADV, `-ctk/-ctv q8_0`.

## Results

Interleaved A/B, 2 reps each, before = `80acfa246` built in a worktree, after = this branch.
Same model, same prompt, greedy, `-c 4096 -ctk q8_0 -ctv q8_0`, MTP draft `n_min 3 / n_max 4`, adaptive.

| | before | after | change |
|---|---|---|---|
| prefill, 1197 tok, ub 512 | 366.9 / 360.5 t/s | 368.4 / 362.7 t/s | 0 (code path unchanged) |
| prefill, 2385 tok, ub 512 -> **ub 2048** | 387.4 / 384.3 t/s | **434.7 / 437.1 t/s** | **+13%** (flag only) |
| decode, no draft (tg 32) | 28.44 / 28.45 t/s | 28.64 / 28.55 t/s | +0.5% (hc.mix node cut, bit-identical output) |
| decode with MTP draft (132 tok) | 27.14 / 27.11 t/s | **45.01 / 45.02 t/s** | **+66%** |
| MTP acceptance as reported | "100%" (artifact) | 71.5% (true) | see lessons |
| MTP draft head Q8_0 -> Q5_K | - | 44.8 / 45.1 t/s, 66% | no gain, not adopted |

Speculative decoding before was *slower* than plain decode on this model (27.1 vs 28.4 t/s): every
verify step paid a ~120 MB host checkpoint of the recurrent state, a graph rebuild + realloc, and on
every partial acceptance a full extra target step to re-decode the accepted tokens. After: 45 t/s,
1.58x plain decode. Earlier in the session the same before-loop measured 37.5 t/s on one run; the
27.1 here is reproducible across both reps and its target GPU time (3.85 s vs ~2.4 s after, perf
logger) confirms the extra target work.

Per-token decode budget is unchanged for plain decode; the levers for that are the quant recipe and
the fused hyper-connection op listed under "Not done".

## What changed on the branch

1. **Timing markers** (`docs/development/timing-markers.md`). `LLAMA_PERF_PHASES=1` prints a host
   wall-clock breakdown of `llama_decode` split prefill/decode; `GGML_VK_PERF_LOGGER_REGIONS=1` groups
   the Vulkan GPU timestamps by model region (`ffn.moe`, `hc.mix`, `gdn.*`, `attn.*`, `ple`, `head`).
   Region tags ride in `ggml_tensor::perf_region`, which took over the old trailing padding.
2. **Graph reuse during speculative verify** (`llama-memory-hybrid-idx.cpp`). The QSA pooled-key
   cache sized a graph input from the exact number of blocks a ubatch completes, which flips between
   1 and 2 across 4-5 token verify steps and forced a rebuild + realloc (~5 ms) on most steps. The
   bound is now derived from the ubatch size; unused rows go to the existing dustbin row. Exact.
3. **Recurrent rollback for qwen4exp** (`llama-arch.cpp`, `build_conv_state_at`). The arch was
   missing from `llm_arch_supports_rs_rollback`, so `n_rs_seq` clamped to 0 and every verify step
   paid a ~120 MB host checkpoint of the recurrent state plus, on rejection, a full re-decode of the
   accepted tokens. Enabling it needed the conv-history write-back to keep the K snapshot slots the
   shared `build_conv_state` keeps; the GDN state, the PLE conv row and the memory planes were
   already generic. Validated with `pocs/rollback-check` (below): state after a rollback differs from
   a never-saw-those-tokens reference by mean 6e-5, two orders of magnitude below the batched-vs-single
   path noise (3e-3).
4. **`hc.mix` cont removal**. The stream-mean chain fed a `ggml_cont` into MULTI_ADD, which already
   takes strided sources. One node per hyper-connection mix, 97 per token. Bit-identical output.
5. **Tried, no gain: draft head lm_head Q8_0 -> Q5_K** (`/tmp/Qwen3.8-Flash-Next-MTP-Q8_0-outQ5K.gguf`).
   The draft's Q8_0 lm_head is 2.9 of its 4.8 ms per draft token and the requant does cut the draft's
   GPU time (2.7 -> 1.9 ms per ubatch), but end-to-end is 44.3 vs 44.4 t/s at 66% vs 72% acceptance:
   the draft is not on the critical path. Keep the Q8 head.
6. `examples/speculative-simple` prints the draft context's perf/phase table; `LLAMA_BUILD_EXAMPLES`
   is on in `build-release`.

## Where the time goes (before)

Decode, 36 ms/token, no draft:

| | ms | |
|---|---|---|
| weight bytes, 4.9 GB/token at ~215 GB/s | ~23 | Q6_K dense backbone 2.2 GB, experts 1.1 GB, Q6_K lm_head 0.52 GB, hc up/down 0.6 GB, F32 router 0.25 GB |
| node latency, ~4100 nodes/token | ~13 | `hc.mix` alone is 1163 nodes/token (28%) |

Prefill, 1197 tokens at ub 512: `ffn.moe` 42% (the `mul_mat_id` is dequant-bound, not
bandwidth-bound: ub 512 -> 2048 only gives +10%), `hc.mix` 12% (memory-bound on 21 MB f32
intermediates, ~10 passes per call), everything attention-related ~4%.

Speculative verify step (~79 ms wall before): 52 ms GPU (a 5-token batch costs 1.45x one token; MoE
scales 1.8x, dense weights are flat), ~7 ms host checkpoint, ~5 ms graph rebuild+realloc. The draft
is 4.8 ms per token, 15% of wall, not the bottleneck.

## Lessons learned

- **Greedy text is not an oracle on this model.** Two single-token steps and the same two tokens as a
  batch differ in recurrent state by ~9% (mean) and in logits by up to 8, with a different top-1, with
  no rollback involved. Layers 0-2 differ by 0.7-2.5% per op because for `n > 1` the Vulkan backend
  quantizes activations to q8_1 (MMVQ) while for `n == 1` on AMD with `k < 2048` it does not; at the
  first attention layer the 512-expert router flips an expert for the second token and the rest is
  chaos. `GGML_VK_DISABLE_MMVQ=1` cuts the disagreement 3x but the remainder (MoE tile path with f16
  B, FA for n > 1) is still large, and spec acceptance does not move (68-71%). Consequences: prefill
  hands decode a state the decode path would not have produced; batched verify rejects drafts the
  single-token path would have produced; KLD measured on batched prefill does not describe decode.
  Validate state changes by comparing tensors/logits against a control, never by diffing text.
- **Check what an acceptance number counts.** `speculative-simple` in checkpoint mode restores the
  checkpoint on partial acceptance and re-decodes the accepted tokens as the next "draft", skipping
  the `n_drafted` update for the rejected round. The reported 100% was that artifact; the true
  acceptance on this prompt is ~69%, and every rejection was costing a full extra target step.
- **Graph reuse needs shape-stable inputs.** Any input tensor whose size depends on where the ubatch
  lands (block boundaries, completed blocks) breaks `can_reuse`; derive sizes from `n_tokens` and pad.
- **Rollback slots are cheap and were already generic**; enabling an arch is a flag plus mirroring the
  K-slot write in any arch-specific state writer. Test it with a state-level comparison.
- **Node count is half of decode on this arch.** ~4100 nodes at 4-5 us each. The hyper-connection mix
  (12 nodes x 97 calls) is the single largest source; a fused op would remove ~15% of all nodes and
  half of `hc.mix`'s prefill traffic. Still the one kernel worth writing.
- **Dense Q6_K costs twice the experts per token.** 2.2 GB vs 1.1 GB. The quant recipe (router
  F32 -> Q8_0, lm_head Q6_K -> Q5_K, backbone Q6_K -> Q5_K) is the largest remaining plain-decode
  lever (~10%) and needs no code; the router exclusion in `llama-quant.cpp:307` needs a switch.
- **Same-input projections are mergeable at load** (hc_down+inject, GDN wqkv+z+beta+alpha, attn
  q+k+v, indexer q+k: ~1.6 ms/token) but the loader has no hook for it; do it in a GGUF rewrite
  alongside the requant.
- **Do not trust sequential runs.** Fixed-draft vs adaptive looked 35% apart in one pair of runs; it
  was GPU state. Interleave arms (your existing rule) and repeat.

## Tools left behind

- `pocs/rollback-check.cpp`: loads the model once, builds a rollback context and a reference
  context, compares recurrent state bytes (`llama_state_seq_get_data`, PARTIAL_ONLY) and logits after
  rollbacks; `trace` mode captures every named tensor for "c2 as 2nd token of a batch" vs "c2 alone"
  through the eval callback and prints per-tensor divergence in graph order. Build line in the header.
- `/tmp/ap-*.sh`: the measurement scripts used here (prompt files `/tmp/ap-prompt*.txt`).

## Not done / next

- Fused hyper-connection op (Vulkan shader + CPU): largest node-count cut.
- Quant recipe requant from `unsloth-q8_0` on scratch (needs ~100 GB freed) + weight merges.
- Set `-ub 2048` (and `-b 2048`) in the preset: +13% prefill measured, no code.
- Server: already handles `COMMON_CONTEXT_SEQ_RM_TYPE_RS` (`server-context.cpp:3250`, `:4105`), so
  it takes the rollback route automatically and only falls back to the checkpoint when a draft is
  longer than `n_rs_seq`. Not measured here.
