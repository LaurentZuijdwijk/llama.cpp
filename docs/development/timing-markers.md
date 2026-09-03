# Timing markers

Two independent breakdowns of a run, both off by default and both split into prefill and decode:

- **host phases** - where the wall clock goes inside `llama_decode`, on any backend
- **GPU regions** - where GPU time goes per part of the model, Vulkan only

A ubatch counts as prefill when it holds more than one token, and as decode otherwise. Speculative
decoding therefore lands in the prefill column, because a draft batch carries several tokens.

## Host phases

```
LLAMA_PERF_PHASES=1 llama-bench -m model.gguf -p 512 -n 64
```

Printed by `llama_perf_context_print`, so `llama-bench`, `llama-completion` and anything else that
calls `common_perf_print` picks it up. Third-party callers can use
`llama_perf_context_print_phases`, which prints nothing when the env var is unset. The table goes
straight to stderr, like the backend perf loggers, so tools that raise the log threshold still show
it.

```
----------------
Phase breakdown:
                          prefill         decode
  ubatches                      1             31
  graph reused                  0             30
  memory update           0.01 ms       0.10 ms   (     10.0 us       3.2 us per ubatch)
  graph build            41.20 ms       0.62 ms   (  41200.0 us      20.0 us per ubatch)
  ...
```

The rows:

| row | what it covers |
|---|---|
| `memory update` | pending KV shifts and copies, before the ubatch split |
| `graph build`   | `model.build_graph`, skipped when the previous graph is reused |
| `graph alloc`   | `ggml_backend_sched_alloc_graph`, also skipped on reuse |
| `set inputs`    | writing the input tensors, including host-side work such as the PLE hash |
| `submit`        | the `graph_compute` call itself |
| `output copy`   | logits, embeddings and sampler readback requests |
| `backend wait`  | `synchronize`, which is where the GPU time lands |

`submit` calls the async compute path, but the Vulkan backend still waits on a fence for most of the
graph before returning, so in practice `submit` already carries most of the GPU time and
`backend wait` picks up the tail. Turning on the Vulkan perf logger adds a synchronous timestamp
readback and moves nearly all of it into `submit`. Read the two rows as one number, and compare
their sum against `prompt eval time` / `eval time` to see how much is host overhead.

## GPU regions

```
GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_REGIONS=1 GGML_VK_PERF_LOGGER_FREQUENCY=100000 \
    llama-bench -m model.gguf -p 512 -n 64
```

`GGML_VK_PERF_LOGGER` collects a GPU timestamp per node. `..._REGIONS` groups those timestamps by
the part of the model that built the node instead of listing every op, and drops the per-op table.
Drop `..._REGIONS` to get both tables. A high `..._FREQUENCY` prints once at the end rather than
once per graph.

```
Vulkan Timings by region:
  prefill  ffn.moe             150254.7 us   61.2 %     4054 nodes      37.1 us/node
  prefill  attn.qsa_index       17067.3 us    7.0 %     1016 nodes      16.8 us/node
  ...
```

Nodes built outside any region show up as `(untagged)`; only `qwen4exp` is tagged so far.

### Tagging a model

`LLM_PERF_REGION("name")` opens a scope: every tensor built while it is alive carries that name, and
the previous region comes back when the scope ends. Nesting works, so a helper can tag itself
without knowing its caller. Inside one scope, `ggml_perf_region_set("name")` switches to a new
region without opening another scope - useful for a straight-line function whose stages share
locals. The names must be string literals: the tag is a bare pointer that has to outlive the graph.

`qwen4exp` uses these regions:

| region | what it covers |
|---|---|
| `embd` | input embedding and the memory/position inputs |
| `ple` | PLE n-gram hash embedding table and its per-layer injection |
| `trunk` | the wide residual outside the per-layer helpers |
| `hc.mix` / `hc.combine` | hyper-connection scatter and gather |
| `attn.qsa_index` | block-sparse indexer: pooled keys, scoring, top-k |
| `attn.proj_qkv` | q/k/v projections, norms, rope |
| `attn.core` | the attention itself, dense or top-k restricted |
| `attn.proj_out` | output gate and output projection |
| `gdn.proj_in` | linear-attention input projections |
| `gdn.conv` | short convolution and its state |
| `gdn.recurrent` | the delta-net scan |
| `gdn.proj_out` | gated norm and output projection |
| `ffn.moe` | routed experts |
| `ffn.shexp` | shared expert and its gate |
| `head` | final norm and lm_head |
| `mtp` | the MTP draft block |

### Cost when off

A region tag is one pointer store per tensor at graph build time, and the tag sits in bytes that
`ggml_tensor` used as padding, so the struct does not grow. Nothing reads the tag unless the Vulkan
perf logger is on.
