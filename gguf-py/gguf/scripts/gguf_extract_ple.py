#!/usr/bin/env python3
"""Pull the joined n-gram table out of a qwen4exp GGUF into its own sidecar file.

The sidecar pairs with --model-ple/-mp: point a backbone build's --model-ple at the
output of this script to test a different PLE precision or source without touching
(or requantizing) the backbone. The loader only ever asks the sidecar for one tensor's
shape, type and file offset - it does not read any KV from it - so the output carries
just enough structure to be a valid GGUF plus the one tensor, copied byte for byte:
no dequantize, no requantize, no loss. Same approach as gguf_split_ple_heads.py.

Only the joined layout (per_layer_token_embd.weight) can be a sidecar today: the
loader's --model-ple path looks up that one name (see llama_model_qwen4exp::
load_arch_tensors in src/models/qwen4exp.cpp). A per-head file is already small
enough to offload without a sidecar, so extracting from one is not supported.
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

logger = logging.getLogger("gguf-extract-ple")

JOINED = "per_layer_token_embd.weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path,
                        help="qwen4exp GGUF with the joined PLE table; pass shard 1 of a "
                             "split file and the rest are picked up")
    parser.add_argument("output", type=Path, help="where to write the sidecar")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.output.exists():
        raise SystemExit(f"{args.output} exists")

    paths = [args.input]
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", args.input.name)
    if m:
        stem, _, total = m.group(1), int(m.group(2)), int(m.group(3))
        paths = [args.input.parent / f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"missing shards: {', '.join(missing)}")
        logger.info(f"reading {total} shards")

    readers = [gguf.GGUFReader(p, "r") for p in paths]
    tensors = [t for r in readers for t in r.tensors]
    joined = next((t for t in tensors if t.name == JOINED), None)
    if joined is None:
        raise ValueError(f"{JOINED} not found - is this the joined layout? "
                          "(a per-head file does not need a sidecar to begin with)")

    logger.info(f"{JOINED}: {joined.tensor_type.name}, {joined.data.shape}, "
                f"{joined.data.nbytes / 2**30:.2f} GiB")

    writer = gguf.GGUFWriter(args.output, arch=readers[0].fields["general.architecture"].contents(),
                             endianess=readers[0].endianess)
    writer.add_tensor_info(JOINED, joined.data.shape, joined.data.dtype, joined.data.nbytes,
                           joined.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.write_tensor_data(joined.data, tensor_endianess=readers[0].endianess)
    writer.close()

    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
