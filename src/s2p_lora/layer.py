"""
S2P-LoRA layer implementation.

Single-slot design: only one adapter topology is active at any training step.

Sequential phase (step k < k_sw):
    lora_A has shape [r, d_out]   — receives W0 @ x ∈ R^{d_out}
    lora_B has shape [d_out, r]
    output: W0 x + B_s A_s (W0 x)

Parallel phase (step k > k_sw):
    lora_A has shape [r, d_in]    — receives x ∈ R^{d_in}   (standard LoRA)
    lora_B has shape [d_out, r]
    output: W0 x + B_p A_p x

At k_sw, a function-preserving core-SVD projection converts
(B_s, A_s) → (B_p, A_p) such that B_p A_p = B_s A_s W0.

For square layers (d_in == d_out): lora_A is overwritten in-place.
For non-square layers: lora_A module is replaced (shape change r×d_out → r×d_in).
    The trainer must update optimizer.param_groups accordingly.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .projection import core_svd_projection


class S2PLoraLinear(nn.Module):
    """
    S2P-LoRA adapter layer wrapping a frozen nn.Linear.

    The layer starts in sequential mode and switches to parallel mode at k_sw.
    Only one set of adapter parameters is active at any time (single-slot design).

    This compact standalone implementation wraps ``nn.Linear`` directly and is
    provided for projection verification and the included trainer integration.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be an nn.Linear instance")
        if r <= 0:
            raise ValueError("r must be positive")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.mode = "sequential"

        d_out, d_in = base_layer.weight.shape
        self.d_out = d_out
        self.d_in = d_in
        if r > min(d_in, d_out):
            raise ValueError(f"r={r} must not exceed min(d_in, d_out)={min(d_in, d_out)}")

        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }

        # Sequential phase: lora_A receives W0 @ x, shape [r, d_out].
        self.lora_A = nn.Linear(d_out, r, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        self.lora_B = nn.Linear(r, d_out, bias=False, **factory_kwargs)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.d_in}, out_features={self.d_out}, r={self.r}, "
            f"scaling={self.scaling:g}, mode={self.mode!r}"
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)

        if self.mode == "sequential":
            adapter_input = F.linear(x, self.base_layer.weight)  # W0 @ x
        else:
            adapter_input = x

        adapter = self.lora_B(self.lora_A(self.dropout(adapter_input))) * self.scaling
        return base + adapter

    # ── Function-preserving switch ───────────────

    @torch.no_grad()
    def switch_to_parallel(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert a sequential adapter to parallel LoRA while preserving its function.

        Computes (B_p, A_p) such that B_p @ A_p = B_s @ A_s @ W0,
        then replaces the adapter parameters in-place (square) or replaces the
        lora_A module (non-square, shape change r×d_out → r×d_in).

        Returns:
            For square layers: []  (in-place overwrite, no module replacement)
            For non-square:    [(old_A_weight, new_A_weight)]
                               so the caller can update optimizer.param_groups.
        """
        if self.mode == "parallel":
            return []

        is_square = self.d_in == self.d_out
        W0 = self.base_layer.weight.detach()  # [d_out, d_in]
        B_s = self.lora_B.weight.detach()  # [d_out, r]
        A_s = self.lora_A.weight.detach()  # [r, d_out]
        device = W0.device
        dtype = W0.dtype

        B_s_f = B_s.float()
        A_s_f = A_s.float()
        W0_f = W0.float()
        B_p, A_p = core_svd_projection(B_s_f, A_s_f, W0_f)

        self.lora_B.weight.copy_(B_p.to(device=device, dtype=dtype))

        replaced: list[tuple[torch.Tensor, torch.Tensor]] = []

        if is_square:
            # In-place: lora_A keeps the same tensor identity
            self.lora_A.weight.copy_(A_p.to(device=device, dtype=dtype))
        else:
            # Shape change r×d_out → r×d_in: must replace module
            old_weight = self.lora_A.weight
            new_A = nn.Linear(self.d_in, self.r, bias=False).to(device=device, dtype=dtype)
            new_A.weight.copy_(A_p.to(device=device, dtype=dtype))
            new_A.weight.requires_grad_(old_weight.requires_grad)
            self.lora_A = new_A
            replaced.append((old_weight, new_A.weight))

        # Clear stale gradients
        if self.lora_A.weight.grad is not None:
            self.lora_A.weight.grad = None
        if self.lora_B.weight.grad is not None:
            self.lora_B.weight.grad = None

        self.mode = "parallel"
        return replaced

    # ── Merge into base weight (for inference) ─────────────────────────────────

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """
        Merge parallel lora_B @ lora_A into base weight.
        Only valid after switching to parallel mode.
        Returns a new nn.Linear with merged weight (no adapter overhead).
        """
        if self.mode != "parallel":
            raise RuntimeError("merge() requires parallel mode; call switch_to_parallel() first.")
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        W_merged = self.base_layer.weight.detach() + (
            self.lora_B.weight.detach() @ self.lora_A.weight.detach() * self.scaling
        )
        merged = nn.Linear(
            self.d_in,
            self.d_out,
            bias=self.base_layer.bias is not None,
        ).to(device=device, dtype=dtype)
        merged.weight.copy_(W_merged.to(device=device, dtype=dtype))
        if self.base_layer.bias is not None:
            merged.bias.copy_(self.base_layer.bias.detach().to(device=device, dtype=dtype))
        merged.train(self.base_layer.training)
        return merged
