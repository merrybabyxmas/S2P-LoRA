#!/usr/bin/env python3
"""Minimal dependency-free S2P-LoRA switch example."""

import torch
from torch import nn

from s2p_lora import S2PLoraLinear


def main() -> None:
    torch.manual_seed(42)
    layer = S2PLoraLinear(
        nn.Linear(64, 48, bias=False),
        r=8,
        lora_alpha=8,
    )
    nn.init.normal_(layer.lora_B.weight, std=0.02)
    inputs = torch.randn(4, 64)

    with torch.no_grad():
        sequential_output = layer(inputs)
        replaced = layer.switch_to_parallel()
        parallel_output = layer(inputs)
        merged_output = layer.merge()(inputs)

    switch_error = torch.linalg.vector_norm(sequential_output - parallel_output)
    merge_error = torch.linalg.vector_norm(parallel_output - merged_output)
    print(f"optimizer parameters replaced: {len(replaced)}")
    print(f"switch error: {switch_error.item():.3e}")
    print(f"merge error:  {merge_error.item():.3e}")


if __name__ == "__main__":
    main()
