"""Tests for the S2P-LoRA function-preserving projection.

Central claim: B_p @ A_p = B_s @ A_s @ W0  (function-preserving switch)

Covers:
- Square matrices (d_in == d_out, e.g. LLaMA q_proj)
- Rectangular, d_out > d_in  (e.g. up-projection)
- Rectangular, d_out < d_in  (e.g. down-projection)
- GQA-like (d_out=1024, d_in=4096, e.g. Mistral k_proj / v_proj)
- Gemma shapes (d_out=4096 d_in=3072 / d_out=3072 d_in=4096)
- Output shapes
- Numerical precision (rel_err < 1e-5 in float32)
- S2PLoraLinear integration (single-slot design)
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from s2p_lora.layer import S2PLoraLinear
from s2p_lora.projection import core_svd_projection

# ── 1. Projection equivalence: B_p @ A_p == B_s @ A_s @ W0 ──────────────────


@pytest.mark.parametrize(
    "d_out,d_in,r",
    [
        (64, 64, 8),  # square
        (128, 64, 8),  # rectangular d_out > d_in
        (64, 128, 8),  # rectangular d_out < d_in
        (1024, 4096, 16),  # GQA-like (Mistral k_proj / v_proj)
        (4096, 3072, 16),  # Gemma q_proj shape
        (3072, 4096, 16),  # Gemma o_proj shape
    ],
)
def test_projection_matrix_equivalence(d_out, d_in, r):
    """B_p @ A_p must equal B_s @ A_s @ W0 up to numerical noise."""
    torch.manual_seed(42)
    W0 = torch.randn(d_out, d_in)
    B_s = torch.randn(d_out, r)
    A_s = torch.randn(r, d_out)

    B_p, A_p = core_svd_projection(B_s, A_s, W0)

    expected = B_s @ A_s @ W0  # [d_out, d_in]
    actual = B_p @ A_p  # [d_out, d_in]

    norm_diff = (expected - actual).norm()
    norm_ref = expected.norm().clamp(min=1e-12)
    rel_err = (norm_diff / norm_ref).item()

    assert rel_err < 1e-5, f"dim ({d_out},{d_in}) r={r}: rel_err={rel_err:.2e} exceeds 1e-5"


def test_projection_preserves_exact_rank_deficiency():
    """Zero singular directions must remain zero instead of receiving an epsilon floor."""
    torch.manual_seed(43)
    W0 = torch.randn(16, 16)
    B_s = torch.randn(16, 4)
    B_s[:, -1] = 0
    A_s = torch.randn(4, 16)
    B_p, A_p = core_svd_projection(B_s, A_s, W0)
    torch.testing.assert_close(B_p @ A_p, B_s @ A_s @ W0, rtol=1e-5, atol=1e-5)


# ── 2. Forward pass equivalence: seq output == par output at switch point ─────


@pytest.mark.parametrize(
    "d_out,d_in,r,batch",
    [
        (64, 64, 8, 5),  # square
        (128, 64, 8, 5),  # rectangular
        (64, 128, 8, 5),  # rectangular
        (1024, 4096, 16, 3),  # GQA-like
    ],
)
def test_forward_equivalence(d_out, d_in, r, batch):
    """
    Sequential forward  h_s = W0 x + B_s A_s (W0 x)
    should equal
    Parallel forward    h_p = W0 x + B_p A_p x
    after projection.
    """
    torch.manual_seed(0)
    W0 = torch.randn(d_out, d_in)
    B_s = torch.randn(d_out, r)
    A_s = torch.randn(r, d_out)
    x = torch.randn(batch, d_in)

    B_p, A_p = core_svd_projection(B_s, A_s, W0)

    # Sequential: base(x) + B_s @ A_s @ base(x)
    # F.linear(input, W) = input @ W.T
    base = F.linear(x, W0)  # [batch, d_out]
    seq_adapter = F.linear(F.linear(base, A_s), B_s)  # [batch, d_out]
    h_seq = base + seq_adapter

    # Parallel: base(x) + B_p @ A_p @ x
    par_adapter = F.linear(F.linear(x, A_p), B_p)  # [batch, d_out]
    h_par = base + par_adapter

    rel_err = ((h_seq - h_par).norm() / h_seq.norm().clamp(min=1e-12)).item()
    assert rel_err < 1e-5, f"dim ({d_out},{d_in}) r={r}: forward rel_err={rel_err:.2e} exceeds 1e-5"


# ── 3. Output shapes ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "d_out,d_in,r",
    [
        (64, 64, 8),
        (128, 64, 8),
        (64, 128, 8),
        (1024, 4096, 128),
        (4096, 3072, 128),
        (3072, 4096, 128),
    ],
)
def test_projection_output_shapes(d_out, d_in, r):
    """B_p must be [d_out, r] and A_p must be [r, d_in]."""
    torch.manual_seed(1)
    W0 = torch.randn(d_out, d_in)
    B_s = torch.randn(d_out, r)
    A_s = torch.randn(r, d_out)

    B_p, A_p = core_svd_projection(B_s, A_s, W0)

    assert B_p.shape == (d_out, r), f"B_p shape wrong: {B_p.shape}"
    assert A_p.shape == (r, d_in), f"A_p shape wrong: {A_p.shape}"


# ── 4. Merge equivalence ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "d_out,d_in,r,batch",
    [
        (64, 64, 8, 7),
        (1024, 4096, 16, 3),
    ],
)
def test_merge_equivalence(d_out, d_in, r, batch):
    """
    After switch, the merged weight W_m = W0 + B_p A_p should satisfy
    W_m @ x == W0 @ x + B_p A_p @ x.
    """
    torch.manual_seed(2)
    W0 = torch.randn(d_out, d_in)
    B_s = torch.randn(d_out, r)
    A_s = torch.randn(r, d_out)
    x = torch.randn(batch, d_in)

    B_p, A_p = core_svd_projection(B_s, A_s, W0)

    W_merged = W0 + B_p @ A_p
    out_merged = F.linear(x, W_merged)
    out_lora = F.linear(x, W0) + F.linear(F.linear(x, A_p), B_p)

    rel_err = ((out_merged - out_lora).norm() / out_lora.norm().clamp(min=1e-12)).item()
    assert rel_err < 1e-6, f"merge rel_err={rel_err:.2e}"


# ── 5. S2PLoraLinear integration (single-slot design) ────────────────────────


def _make_layer(d_in, d_out, r=8):
    base = nn.Linear(d_in, d_out, bias=False)
    return S2PLoraLinear(base_layer=base, r=r, lora_alpha=float(r))


def _initialize_nonzero_adapter(layer):
    """Make layer-level equivalence tests exercise a nonzero adapter path."""
    nn.init.normal_(layer.lora_B.weight, std=0.02)


@pytest.mark.parametrize(
    "d_in,d_out,r",
    [
        (64, 64, 8),  # square
        (128, 64, 8),  # non-square d_in > d_out
        (64, 128, 8),  # non-square d_out > d_in
    ],
)
def test_layer_forward_function_preserving(d_in, d_out, r):
    """
    Forward output must be identical before and after the projection switch:
      Before: h = W0 x + B_s A_s (W0 x)
      After:  h = W0 x + B_p A_p x
    """
    torch.manual_seed(3)
    layer = _make_layer(d_in, d_out, r)
    _initialize_nonzero_adapter(layer)
    layer.eval()
    x = torch.randn(5, d_in)

    with torch.no_grad():
        h_seq = layer(x)
        layer.switch_to_parallel()
        h_par = layer(x)

    rel_err = ((h_seq - h_par).norm() / h_seq.norm().clamp(min=1e-12)).item()
    assert rel_err < 1e-4, (
        f"S2PLoraLinear ({d_in}->{d_out}) forward rel_err={rel_err:.2e} after switch"
    )


@pytest.mark.parametrize(
    "d_in,d_out,r",
    [
        (64, 64, 8),
        (128, 64, 8),
        (64, 128, 8),
    ],
)
def test_layer_lora_a_shape_after_switch(d_in, d_out, r):
    """After switch, lora_A must have shape [r, d_in] (parallel-phase shape)."""
    layer = _make_layer(d_in, d_out, r)
    with torch.no_grad():
        layer.switch_to_parallel()
    assert layer.lora_A.weight.shape == (r, d_in), (
        f"Expected lora_A shape ({r},{d_in}), got {layer.lora_A.weight.shape}"
    )


def test_layer_square_returns_empty_replaced_params():
    """Square layers: switch returns [] (in-place overwrite)."""
    layer = _make_layer(d_in=64, d_out=64, r=8)
    with torch.no_grad():
        replaced = layer.switch_to_parallel()
    assert replaced == [], f"Square layer should return [], got {replaced}"


def test_layer_nonsquare_returns_replaced_params():
    """Non-square layers: switch returns [(old_weight, new_weight)]."""
    layer = _make_layer(d_in=128, d_out=64, r=8)
    old_lora_a = layer.lora_A.weight
    with torch.no_grad():
        replaced = layer.switch_to_parallel()
    assert len(replaced) == 1, f"Expected 1 pair, got {len(replaced)}"
    old_p, new_p = replaced[0]
    assert old_p is old_lora_a, "old_p should be the original lora_A weight"
    assert old_p.shape != new_p.shape, "old and new param shapes must differ"


def test_layer_mode_after_switch():
    """After switch, layer.mode must be 'parallel'."""
    layer = _make_layer(64, 64, r=8)
    assert layer.mode == "sequential"
    with torch.no_grad():
        layer.switch_to_parallel()
    assert layer.mode == "parallel"


def test_layer_single_slot_invariant_nonsquare():
    """
    Single-slot invariant: after switch on a non-square layer,
    lora_A has shape (r, d_in) — sequential shape is gone.
    """
    d_in, d_out, r = 128, 64, 8
    layer = _make_layer(d_in, d_out, r)
    assert layer.lora_A.weight.shape == (r, d_out)  # sequential phase

    with torch.no_grad():
        layer.switch_to_parallel()

    assert layer.lora_A.weight.shape == (r, d_in)  # parallel phase
    assert layer.mode == "parallel"


def test_layer_merge():
    """merge() must produce a plain nn.Linear with W0 + B_p A_p fused."""
    d_in, d_out, r = 64, 64, 8
    torch.manual_seed(7)
    layer = _make_layer(d_in, d_out, r)
    _initialize_nonzero_adapter(layer)
    x = torch.randn(4, d_in)

    with torch.no_grad():
        layer.switch_to_parallel()
        h_adapter = layer(x)
        merged = layer.merge()
        h_merged = merged(x)

    rel_err = ((h_adapter - h_merged).norm() / h_adapter.norm().clamp(min=1e-12)).item()
    assert rel_err < 1e-6, f"merge() rel_err={rel_err:.2e}"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_bfloat16_nonsquare_switch_forward_and_merge(device):
    """Regression test for BF16 rectangular projections and merged placement."""
    torch.manual_seed(9)
    base = nn.Linear(128, 64, bias=True).to(device=device, dtype=torch.bfloat16)
    layer = S2PLoraLinear(base, r=8, lora_alpha=8.0)
    _initialize_nonzero_adapter(layer)
    layer.eval()
    x = torch.randn(3, 128, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        sequential_output = layer(x)
        layer.switch_to_parallel()
        parallel_output = layer(x)
        merged = layer.merge()
        merged_output = merged(x)

    assert layer.lora_A.weight.dtype == torch.bfloat16
    assert layer.lora_A.weight.device == base.weight.device
    assert merged.weight.dtype == torch.bfloat16
    assert merged.weight.device == base.weight.device

    switch_rel_err = (
        (sequential_output.float() - parallel_output.float()).norm()
        / sequential_output.float().norm().clamp(min=1e-12)
    ).item()
    merge_rel_err = (
        (parallel_output.float() - merged_output.float()).norm()
        / parallel_output.float().norm().clamp(min=1e-12)
    ).item()
    assert switch_rel_err < 2e-2
    assert merge_rel_err < 2e-2
