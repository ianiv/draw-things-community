#!/usr/bin/env python3
"""Convert a HyperFlow H3 Flash GGUF transformer into a safetensors file Draw Things can import.

HyperFlow H3 Flash (VideoRebirthAI/hyperflow-h3-flash) is a compressed MiniMax H3: it drops a few
transformer blocks, stores some linears as low-rank `down` / `up` pairs and adds an endpoint
time embedder. Its GGUF files renumber the kept blocks without gaps; Draw Things needs the
original block indices (they select per-block FP16 scale factors), so this script restores them
from the `dropped_blocks` list of the model's config.json.

Quantized tensors (Q8_0 and friends) are dequantized to float16; BF16 and F32 tensors are written
as float32 so nothing underflows. Tensors are streamed one at a time, so memory stays low.

Usage:
  python hyperflow_gguf_to_safetensors.py fl2va-Q8_0.gguf out/hyperflow_h3_flash_fl2va.safetensors \
      --config fl2va_int8_convrot/config.json

Name the output with "ref2va" in it for the Ref2VA partition: the importer picks the sampler
modifier from the file name. Requires `pip install gguf numpy`.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize

PREFIX = "model.diffusion_model."
BLOCK = re.compile(r"^blocks\.(\d+)\.(.*)$")
NUM_LAYERS = 50
# Known partitions, used when no config.json is given.
DEFAULT_DROPPED = {"fl2va": [6, 21, 22], "ref2va": [9]}


def dropped_blocks(args: argparse.Namespace) -> list[int]:
    if args.config:
        config = json.loads(Path(args.config).read_text())
        return sorted(int(i) for i in config.get("dropped_blocks") or [])
    if args.dropped is not None:
        return sorted(int(i) for i in args.dropped.split(",") if i.strip())
    name = Path(args.gguf).name.lower()
    for partition, dropped in DEFAULT_DROPPED.items():
        if partition in name:
            print(f"Assuming {partition} dropped blocks {dropped}", file=sys.stderr)
            return dropped
    raise SystemExit("Pass --config <config.json> or --dropped 6,21,22")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("gguf")
    parser.add_argument("output")
    parser.add_argument("--config", help="config.json of the same partition (for dropped_blocks)")
    parser.add_argument("--dropped", help="comma-separated dropped block indices")
    args = parser.parse_args()

    dropped = dropped_blocks(args)
    kept = [i for i in range(NUM_LAYERS) if i not in dropped]
    reader = GGUFReader(args.gguf)

    plan = []  # (name, tensor, dtype, shape)
    positions = set()
    for tensor in reader.tensors:
        name = tensor.name.removeprefix(PREFIX)
        match = BLOCK.match(name)
        if match:
            position = int(match.group(1))
            positions.add(position)
            name = f"blocks.{kept[position]}.{match.group(2)}"
        shape = [int(size) for size in reversed(tensor.shape)]
        if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.BF16):
            dtype = "F32"
        else:
            dtype = "F16"
        plan.append((name, tensor, dtype, shape))
    if positions and max(positions) + 1 != len(kept):
        raise SystemExit(
            f"GGUF has {max(positions) + 1} blocks but config keeps {len(kept)}; wrong config.json?"
        )

    header = {}
    offset = 0
    for name, _, dtype, shape in plan:
        size = int(np.prod(shape)) * (4 if dtype == "F32" else 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {
        "format": "pt",
        "source": Path(args.gguf).name,
        "dropped_blocks": json.dumps(dropped),
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for index, (name, tensor, dtype, shape) in enumerate(plan):
            if tensor.tensor_type == GGMLQuantizationType.F32:
                values = np.asarray(tensor.data, dtype=np.float32)
            else:
                values = dequantize(tensor.data, tensor.tensor_type)
            values = np.ascontiguousarray(values.reshape(shape), dtype=np.float32 if dtype == "F32" else np.float16)
            if dtype == "F16" and not np.isfinite(values).all():
                raise SystemExit(f"{name} overflows float16")
            out.write(values.tobytes())
            if index % 50 == 0:
                print(f"{index + 1}/{len(plan)} {name} {shape}", file=sys.stderr)
    print(f"Wrote {args.output} ({offset / 1e9:.1f} GB, {len(plan)} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
