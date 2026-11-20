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
        self._mode = "sequential"
        self.register_buffer(
            "_s2p_parallel", torch.tensor(False, device=base_layer.weight.device)
        )

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

    @property
    def mode(self) -> str:
        return self._mode

    def _set_mode(self, mode: str) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Select the factor shape before loading or copying weights."""
        if mode not in ("sequential", "parallel"):
            raise ValueError(f"Invalid adapter mode: {mode!r}")
        if mode == self.mode:
            return []

        width = self.d_in if mode == "parallel" else self.d_out
        replaced = []
        if width != self.lora_A.in_features:
            old = self.lora_A.weight
            replacement = nn.Linear(width, self.r, bias=False, device=old.device, dtype=old.dtype)
            replacement.weight.requires_grad_(old.requires_grad)
            replacement.train(self.training)
            self.lora_A = replacement
            replaced.append((old, replacement.weight))
        self._mode = mode
        self._s2p_parallel.fill_(mode == "parallel")
        return replaced

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        phase_key = prefix + "_s2p_parallel"
        phase = state_dict.get(phase_key)
        if phase is not None:
            if phase.shape != () or phase.dtype != torch.bool:
                error_msgs.append(f"Invalid S2P phase marker for {prefix}")
                return
            self._set_mode("parallel" if phase.item() else "sequential")
        else:
            # Older raw state dicts identify the phase by shape only on rectangular layers.
            factor = state_dict.get(prefix + "lora_A.weight")
            if factor is not None and self.d_in != self.d_out:
                if factor.shape == (self.r, self.d_in):
                    self._set_mode("parallel")
                elif factor.shape == (self.r, self.d_out):
                    self._set_mode("sequential")
            state_dict = {**state_dict, phase_key: self._s2p_parallel}
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)

        if self.mode == "sequential":
            adapter_input = (
                base if self.base_layer.bias is None else F.linear(x, self.base_layer.weight)
            )
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

        return self._apply_parallel_factors(*self._parallel_factors())

    @torch.no_grad()
    def _parallel_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        W0 = self.base_layer.weight.detach()  # [d_out, d_in]
        B_s = self.lora_B.weight.detach()  # [d_out, r]
        A_s = self.lora_A.weight.detach()  # [r, d_out]
        work_dtype = torch.float64 if W0.dtype == torch.float64 else torch.float32
        B_s_f = B_s.to(work_dtype)
        A_s_f = A_s.to(work_dtype)
        W0_f = W0.to(work_dtype)
        B_p, A_p = core_svd_projection(B_s_f, A_s_f, W0_f)

        # A zero column of stored B permits any matching row of A without
        # changing the represented update. Keep those directions trainable.
        B_p = B_p.to(self.lora_B.weight.dtype)
        empty = ~B_p.ne(0).any(dim=0)
        basis = torch.eye(self.r, self.d_in, device=A_p.device, dtype=A_p.dtype)
        A_p = torch.where(empty[:, None], basis, A_p).to(self.lora_A.weight.dtype)
        if not torch.isfinite(A_p).all() or not torch.isfinite(B_p).all():
            raise ValueError("S2P projection produced non-finite factors")

        return B_p, A_p

    @torch.no_grad()
    def _apply_parallel_factors(
        self, B_p: torch.Tensor, A_p: torch.Tensor
    ) -> list[tuple[nn.Parameter, nn.Parameter]]:
        replaced = self._set_mode("parallel")
        self.lora_B.weight.copy_(B_p)
        self.lora_A.weight.copy_(A_p)

        # Clear stale gradients
        if self.lora_A.weight.grad is not None:
            self.lora_A.weight.grad = None
        if self.lora_B.weight.grad is not None:
            self.lora_B.weight.grad = None

        return replaced

    # ── Merge into base weight (for inference) ─────────────────────────────────

    @torch.no_grad()
    def merge(self, *, safe_merge: bool = True) -> nn.Linear:
        """
        Merge parallel lora_B @ lora_A into base weight.
        Only valid after switching to parallel mode.
        Returns a new nn.Linear with merged weight (no adapter overhead).
        """
        if self.mode != "parallel":
            raise RuntimeError("merge() requires parallel mode; call switch_to_parallel() first.")
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        W_merged = self.base_layer.weight.detach().to(work_dtype) + (
            self.lora_B.weight.detach().to(work_dtype)
            @ self.lora_A.weight.detach().to(work_dtype)
            * self.scaling
        )
        W_merged = W_merged.to(dtype)
        if safe_merge and not torch.isfinite(W_merged).all():
            raise ValueError("Cannot merge non-finite S2P-LoRA weights")
        merged = nn.Linear(
            self.d_in,
            self.d_out,
            bias=self.base_layer.bias is not None,
            device=device,
            dtype=dtype,
        )
        merged.weight.copy_(W_merged)
        merged.weight.requires_grad_(self.base_layer.weight.requires_grad)
        if self.base_layer.bias is not None:
            merged.bias.copy_(self.base_layer.bias)
            merged.bias.requires_grad_(self.base_layer.bias.requires_grad)
        merged.train(self.base_layer.training)
        return merged
