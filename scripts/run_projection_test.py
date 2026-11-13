"""Standalone correctness check for the S2P-LoRA projection and layer switch.

Run from the submission/ directory:
    python3 scripts/run_projection_test.py

Expected output: all cases PASS with precision-appropriate tolerances.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch import nn

from s2p_lora.layer import S2PLoraLinear
from s2p_lora.projection import core_svd_projection


def check(label, rel_err, tol=1e-5):
    status = "PASS" if rel_err < tol else "FAIL"
    print(f"  [{status}] {label:50s}  rel_err={rel_err:.2e}  (tol={tol:.0e})")
    return status == "PASS"


def test_projection(d_out, d_in, r, label):
    torch.manual_seed(42)
    W0 = torch.randn(d_out, d_in)
    B_s = torch.randn(d_out, r)
    A_s = torch.randn(r, d_out)

    B_p, A_p = core_svd_projection(B_s, A_s, W0)

    expected = B_s @ A_s @ W0
    actual = B_p @ A_p

    rel_err = ((expected - actual).norm() / expected.norm().clamp(min=1e-12)).item()
    return check(label, rel_err)


def test_forward(d_out, d_in, r, batch, label):
    torch.manual_seed(0)
    W0 = torch.randn(d_out, d_in)
    B_s = torch.randn(d_out, r)
    A_s = torch.randn(r, d_out)
    x = torch.randn(batch, d_in)

    B_p, A_p = core_svd_projection(B_s, A_s, W0)

    base = F.linear(x, W0)
    h_seq = base + F.linear(F.linear(base, A_s), B_s)
    h_par = base + F.linear(F.linear(x, A_p), B_p)

    rel_err = ((h_seq - h_par).norm() / h_seq.norm().clamp(min=1e-12)).item()
    return check(label, rel_err)


def test_layer_switch(d_in, d_out, r, label):
    torch.manual_seed(3)
    base = nn.Linear(d_in, d_out, bias=False)
    layer = S2PLoraLinear(base_layer=base, r=r, lora_alpha=float(r))
    nn.init.normal_(layer.lora_B.weight, std=0.02)
    layer.eval()
    x = torch.randn(5, d_in)

    with torch.no_grad():
        h_seq = layer(x)
        layer.switch_to_parallel()
        h_par = layer(x)

    rel_err = ((h_seq - h_par).norm() / h_seq.norm().clamp(min=1e-12)).item()
    return check(label, rel_err, tol=1e-4)


def main():
    print("=" * 70)
    print("S2P-LoRA: Function-Preserving Projection Correctness Check")
    print("Claim: B_p @ A_p = B_s @ A_s @ W0  (function-preserving switch)")
    print("=" * 70)

    all_pass = True

    print("\n[1] Matrix equivalence: B_p @ A_p == B_s @ A_s @ W0")
    cases = [
        (64, 64, 8, "square (64x64, r=8)"),
        (128, 64, 8, "rect d_out>d_in (128x64, r=8)"),
        (64, 128, 8, "rect d_out<d_in (64x128, r=8)"),
        (1024, 4096, 16, "GQA k/v_proj (1024x4096, r=16)"),
        (4096, 3072, 16, "Gemma q_proj (4096x3072, r=16)"),
        (3072, 4096, 16, "Gemma o_proj (3072x4096, r=16)"),
    ]
    for d_out, d_in, r, label in cases:
        all_pass &= test_projection(d_out, d_in, r, label)

    print("\n[2] Forward equivalence: h_seq == h_par at switch point")
    fwd_cases = [
        (64, 64, 8, 5, "square (64x64, r=8, batch=5)"),
        (128, 64, 8, 5, "rect (128x64, r=8, batch=5)"),
        (64, 128, 8, 5, "rect (64x128, r=8, batch=5)"),
        (1024, 4096, 16, 3, "GQA-like (1024x4096, r=16, batch=3)"),
    ]
    for d_out, d_in, r, batch, label in fwd_cases:
        all_pass &= test_forward(d_out, d_in, r, batch, label)

    print("\n[3] S2PLoraLinear layer switch (single-slot design)")
    layer_cases = [
        (64, 64, 8, "square (64->64, r=8)"),
        (128, 64, 8, "non-square d_in>d_out (128->64, r=8)"),
        (64, 128, 8, "non-square d_out>d_in (64->128, r=8)"),
    ]
    for d_in, d_out, r, label in layer_cases:
        all_pass &= test_layer_switch(d_in, d_out, r, label)

    print("\n[4] Optimizer param replacement check (non-square)")
    base = nn.Linear(128, 64, bias=False)
    layer = S2PLoraLinear(base_layer=base, r=8, lora_alpha=8.0)
    old_weight = layer.lora_A.weight
    with torch.no_grad():
        replaced = layer.switch_to_parallel()
    ok = len(replaced) == 1 and replaced[0][0] is old_weight
    all_pass &= ok
    print(
        f"  [{'PASS' if ok else 'FAIL'}] non-square returns (old, new) param pair  "
        f"len={len(replaced)}, shapes: {replaced[0][0].shape} -> {replaced[0][1].shape}"
    )

    print("\n[5] Square returns empty replaced list")
    base2 = nn.Linear(64, 64, bias=False)
    layer2 = S2PLoraLinear(base_layer=base2, r=8, lora_alpha=8.0)
    with torch.no_grad():
        replaced2 = layer2.switch_to_parallel()
    ok2 = replaced2 == []
    all_pass &= ok2
    print(f"  [{'PASS' if ok2 else 'FAIL'}] square returns []  got={replaced2}")

    print("\n[6] BF16 non-square switch and merge placement")
    torch.manual_seed(11)
    base3 = nn.Linear(128, 64, bias=True).to(dtype=torch.bfloat16)
    layer3 = S2PLoraLinear(base3, r=8, lora_alpha=8.0)
    nn.init.normal_(layer3.lora_B.weight, std=0.02)
    x3 = torch.randn(3, 128, dtype=torch.bfloat16)
    with torch.no_grad():
        h3_seq = layer3(x3)
        layer3.switch_to_parallel()
        h3_par = layer3(x3)
        merged3 = layer3.merge()
        h3_merged = merged3(x3)
    switch_err = (
        (h3_seq.float() - h3_par.float()).norm() / h3_seq.float().norm().clamp(min=1e-12)
    ).item()
    merge_err = (
        (h3_par.float() - h3_merged.float()).norm() / h3_par.float().norm().clamp(min=1e-12)
    ).item()
    dtype_ok = (
        layer3.lora_A.weight.dtype == torch.bfloat16
        and merged3.weight.dtype == torch.bfloat16
        and merged3.weight.device == base3.weight.device
    )
    bf16_ok = dtype_ok and switch_err < 2e-2 and merge_err < 2e-2
    all_pass &= bf16_ok
    print(
        f"  [{'PASS' if bf16_ok else 'FAIL'}] BF16 rectangular forward/merge  "
        f"switch_err={switch_err:.2e}, merge_err={merge_err:.2e}, "
        f"dtype={layer3.lora_A.weight.dtype}"
    )

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — check output above")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
