#!/usr/bin/env python3
"""Split a qwen4exp GGUF into a backbone shard and a swappable PLE shard.

Produces a stock two-shard GGUF split:

    <out>-00001-of-00002.gguf   everything except per_layer_token_embd.weight
    <out>-00002-of-00002.gguf   only per_layer_token_embd.weight

Both are loaded by unmodified llama.cpp - point `-m` at shard 1 and the second is
picked up automatically. No fork, no extra flag. This is the mainline-compatible
alternative to --model-ple.

Why the PLE shard is swappable
------------------------------
llama_model_loader validates a split on exactly four things (see the n_split block
in src/llama-model-loader.cpp):

  * `split.count` matches the number of files found by the glob,
  * every shard's own `split.no` equals its index,
  * no tensor name appears in two shards,
  * the total tensor count equals `split.tensors.count`.

It reads `split.count` / `split.tensors.count` from shard 1 only, and never compares
tensor *types* or per-shard byte counts across shards - each shard's quantization
type is taken from its own tensor info. GGUFWriter also puts the whole KV block in
shard 1 and gives later shards nothing but the three split keys, so the PLE shard
carries no model-specific metadata whatsoever.

Together that means one PLE shard file can be paired with the backbone of *any*
tier, at any of the PLE-legal precisions, just by naming it `-00002-of-00002.gguf`
alongside the backbone. The table is byte-identical across tiers built from the same
source anyway, so shipping it once instead of once per tier also removes a large
amount of duplication.

`per_layer_token_embd.weight` has ne0 = 160, which is not a multiple of 256, so only
32-block types are legal on it: IQ4_NL (4.5 bpw), Q5_0 (5.5), Q5_1 (6.0), Q8_0 (8.5).

Bytes are copied verbatim - no dequantize, no requantize, no loss. Only the joined
layout is handled; a per-head file has its table spread over ple_ngram_embd.N.weight
and does not need this treatment.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf  # noqa: E402

logger = logging.getLogger("gguf-split-ple-shard")

JOINED = "per_layer_token_embd.weight"


def shard_paths(prefix: Path) -> list[Path]:
    return [prefix.parent / f"{prefix.name}-{i:05d}-of-00002.gguf" for i in (1, 2)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path,
                        help="qwen4exp GGUF with the joined PLE table; pass shard 1 of a "
                             "split file and the rest are picked up")
    parser.add_argument("output_prefix", type=Path,
                        help="output prefix; -00001-of-00002.gguf and -00002-of-00002.gguf "
                             "are appended")
    parser.add_argument("--tensor", default=JOINED,
                        help=f"tensor to isolate into shard 2 (default: {JOINED}). "
                             "Overridable mainly so the mechanism can be exercised on a "
                             "small model without a PLE table.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    outs = shard_paths(args.output_prefix)
    for p in outs:
        if p.exists():
            raise SystemExit(f"{p} exists")

    # follow an existing split on the input side
    paths = [args.input]
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", args.input.name)
    if m:
        stem, total = m.group(1), int(m.group(3))
        paths = [args.input.parent / f"{stem}-{i:05d}-of-{total:05d}.gguf"
                 for i in range(1, total + 1)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"missing shards: {', '.join(missing)}")
        logger.info(f"reading {total} input shards")

    readers = [gguf.GGUFReader(p, "r") for p in paths]
    tensors = [t for r in readers for t in r.tensors]

    target = args.tensor
    ple = next((t for t in tensors if t.name == target), None)
    if ple is None:
        raise SystemExit(f"{target} not found - if you expected the joined PLE layout, "
                         "note a per-head file spreads it over ple_ngram_embd.N.weight "
                         "and does not need splitting")
    rest = [t for t in tensors if t.name != target]

    logger.info(f"backbone : {len(rest)} tensors")
    logger.info(f"isolated : {target}\n"
                f"           {ple.tensor_type.name}, {tuple(ple.data.shape)}, "
                f"{ple.data.nbytes / 2**30:.2f} GiB")

    arch = readers[0].fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(args.output_prefix, arch=arch,
                             endianess=readers[0].endianess, use_temp_file=False)

    # carry over every KV field from the source except the split bookkeeping, which
    # GGUFWriter regenerates for the new two-shard layout
    skip = {"general.architecture", "split.no", "split.count", "split.tensors.count"}
    for key, field in readers[0].fields.items():
        if key in skip:
            continue
        writer.add_key_value(key, field.contents(), field.types[0],
                             sub_type=field.types[-1] if len(field.types) > 1 else None)

    # shard 1: everything but the table
    for t in rest:
        writer.add_tensor_info(t.name, tuple(t.data.shape), t.data.dtype,
                               t.data.nbytes, t.tensor_type)

    # shard 2: the table alone. starting a new dict here is what forces the split -
    # add_tensor_info only rolls over on its own when a size/count threshold is hit.
    writer.tensors.append({})
    writer.add_tensor_info(ple.name, tuple(ple.data.shape), ple.data.dtype,
                           ple.data.nbytes, ple.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for t in rest:
        writer.write_tensor_data(t.data, tensor_endianess=readers[0].endianess)
    writer.write_tensor_data(ple.data, tensor_endianess=readers[0].endianess)
    writer.close()

    for p in outs:
        logger.info(f"wrote {p}  ({p.stat().st_size / 2**30:.2f} GiB)")
    logger.info(f"run with: llama-cli -m {outs[0]}")
    logger.info(f"swap the table by replacing {outs[1].name} with another PLE shard")


if __name__ == "__main__":
    main()
