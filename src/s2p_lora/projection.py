"""S2P-LoRA function-preserving projection.

Converts a trained sequential adapter (B_s, A_s) into a parallel LoRA
adapter (B_p, A_p) such that B_p @ A_p = B_s @ A_s @ W0.

The conversion is function-preserving at the switch point:
    sequential output: W0 x + B_s A_s (W0 x)
    parallel  output:  W0 x + B_p A_p x
    equality holds because B_p A_p = B_s A_s W0.

Works for both square (d_in == d_out) and rectangular weight matrices.
The rectangular case arises in grouped-query attention projections,
e.g. k_proj, v_proj in Mistral-7B and all projections in Gemma-7B.
"""

import torch


@torch.no_grad()
def core_svd_projection(
    B_s: torch.Tensor,
    A_s: torch.Tensor,
    W0: torch.Tensor,
    eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function-preserving core-SVD projection from sequential to parallel factors.

    Args:
        B_s: Sequential lora_B weight, shape [d_out, r].
             (nn.Linear weight convention: [out_features, in_features])
        A_s: Sequential lora_A weight, shape [r, d_out].
             Receives W0 @ x during the sequential phase.
        W0:  Frozen base weight, shape [d_out, d_in].
        eps: Optional singular-value floor. The default ``0`` preserves exact
             zero singular values and therefore the represented function.

    Returns:
        B_p: Parallel lora_B weight, shape [d_out, r].
        A_p: Parallel lora_A weight, shape [r, d_in].
             Receives x directly during the parallel phase.

    Guarantee:
        B_p @ A_p = B_s @ A_s @ W0   (up to float32 numerical noise, rel_err < 1e-5)

    Algorithm (QR + low-rank SVD, O(r·d_in·d_out) + O(r²·d_in) + O(r³)):
        1. X  = A_s @ W0              [r, d_in]   (project sequential A into input space)
        2. QR decompose B_s and X.T   [(r,r) cores]
        3. SVD of r×r core C = R_B R_X^T   [U, S, Vh]
        4. B_p = Q_B U √S,  A_p = √S Vh Q_X^T   (balanced split)

    The measured speedup over full-weight SVD is 38×–505× for the model
    shapes reported in the paper.
    """
    if B_s.ndim != 2 or A_s.ndim != 2 or W0.ndim != 2:
        raise ValueError("B_s, A_s, and W0 must all be rank-2 tensors")
    d_out, rank = B_s.shape
    if W0.shape[0] != d_out:
        raise ValueError(f"B_s and W0 disagree on d_out: {d_out} != {W0.shape[0]}")
    if A_s.shape != (rank, d_out):
        raise ValueError(
            f"Expected A_s shape [rank, d_out] = ({rank}, {d_out}), got {tuple(A_s.shape)}"
        )
    if rank > min(W0.shape):
        raise ValueError(f"rank={rank} must not exceed min(d_out, d_in)={min(W0.shape)}")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    X = A_s @ W0  # [r, d_in]

    Q_B, R_B = torch.linalg.qr(B_s, mode="reduced")  # (d_out,r), (r,r)
    Q_X, R_X = torch.linalg.qr(X.T, mode="reduced")  # (d_in,r),  (r,r)

    C = R_B @ R_X.T  # [r, r]
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)

    U_M = Q_B @ U  # [d_out, r]
    Vh_M = Vh @ Q_X.T  # [r, d_in]

    sqrt_S = torch.sqrt(S.clamp_min(eps))
    B_p = U_M * sqrt_S.unsqueeze(0)  # [d_out, r]
    A_p = sqrt_S.unsqueeze(1) * Vh_M  # [r, d_in]

    return B_p, A_p
