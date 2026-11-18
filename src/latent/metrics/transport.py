"""Transport metrics: Transport Linearity Score (τ) and Transport Asymmetry Index (TAI).

Implements §2.2 and §2.4 of the BCCT framework. Fits linear (Procrustes) and
nonlinear (2-layer MLP) transport maps between representation spaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransportResult:
    """Result of a directional transport evaluation.

    Attributes:
        perf_linear: Downstream performance using linear (Procrustes) map.
        perf_best: Downstream performance using best (MLP) map.
        tau: Transport Linearity Score = perf_linear / perf_best.
        map_linear_weights: Learned Procrustes rotation matrix.
        converged: Whether the MLP training converged (final loss < 2× best loss).
    """

    perf_linear: float
    perf_best: float
    tau: float
    map_linear_weights: Tensor
    converged: bool = True


@dataclass(frozen=True)
class BidirectionalTransportResult:
    """Result of bidirectional transport evaluation.

    Attributes:
        a_to_b: Transport result in A → B direction.
        b_to_a: Transport result in B → A direction.
        tai: Transport Asymmetry Index.
        asymmetry_direction: Which direction performs better ("A→B" or "B→A").
    """

    a_to_b: TransportResult
    b_to_a: TransportResult
    tai: float
    asymmetry_direction: str


class _TransportMLP(nn.Module):
    """Standard 2-layer MLP for nonlinear transport (M_best).

    Architecture: Linear(d_in, 2*d_out) → ReLU → Linear(2*d_out, d_out).
    This is the standardized architecture from §2.2 of the BCCT framework.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        hidden = max(2 * d_out, 256)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _fit_procrustes(source: Tensor, target: Tensor) -> Tensor:
    """Fit orthogonal Procrustes alignment: find Q* = argmin ||target - source @ Q||_F.

    When dimensions differ, uses the pseudo-inverse approach:
    first projects source to target dimension via least-squares, then
    extracts the orthogonal component via SVD.

    Args:
        source: Source representations, shape (N, D_source).
        target: Target representations, shape (N, D_target).

    Returns:
        Optimal linear map W, shape (D_source, D_target).
    """
    # Center both
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    # Cross-covariance
    m = source_c.T @ target_c  # (D_source, D_target)

    # SVD for Procrustes solution
    u, _, vh = torch.linalg.svd(m, full_matrices=False)  # type: ignore[reportUnknownMemberType]

    # Optimal orthogonal map (possibly rectangular)
    q = u @ vh  # type: ignore[reportUnknownVariableType]

    return q  # type: ignore[reportUnknownVariableType]


@torch.no_grad()
def _evaluate_linear_transport(
    source_train: Tensor,
    target_train: Tensor,
    source_test: Tensor,
    target_test: Tensor,
) -> tuple[float, Tensor]:
    """Fit Procrustes and evaluate reconstruction quality.

    Args:
        source_train: Training source features, shape (N_train, D_s).
        target_train: Training target features, shape (N_train, D_t).
        source_test: Test source features, shape (N_test, D_s).
        target_test: Test target features, shape (N_test, D_t).

    Returns:
        Tuple of (r2_score, weight_matrix).
    """
    w = _fit_procrustes(source_train, target_train)

    # Predict
    source_mean = source_train.mean(dim=0, keepdim=True)
    target_mean = target_train.mean(dim=0, keepdim=True)
    pred = (source_test - source_mean) @ w + target_mean

    # Per-dimension R² (avoids collapse in high-dimensional spaces)
    ss_res_d = ((target_test - pred) ** 2).sum(dim=0)
    ss_tot_d = ((target_test - target_test.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)

    if ss_tot_d.max() < 1e-10:
        return 1.0, w
    r2_d = 1.0 - ss_res_d / ss_tot_d.clamp(min=1e-10)
    r2 = r2_d.clamp(min=0.0).mean().item()

    return r2, w


def _evaluate_mlp_transport(
    source_train: Tensor,
    target_train: Tensor,
    source_test: Tensor,
    target_test: Tensor,
    n_steps: int = 3000,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[float, bool]:
    """Fit 2-layer MLP and evaluate reconstruction quality.

    Args:
        source_train: Training source features, shape (N_train, D_s).
        target_train: Training target features, shape (N_train, D_t).
        source_test: Test source features, shape (N_test, D_s).
        target_test: Test target features, shape (N_test, D_t).
        n_steps: Number of Adam optimization steps.
        lr: Learning rate.
        seed: Random seed for weight initialization.

    Returns:
        Tuple of (per-dimension R² score on test set, whether training converged).
    """
    torch.manual_seed(seed)  # type: ignore[arg-type]
    device = source_train.device

    d_in = source_train.shape[1]
    d_out = target_train.shape[1]
    model = _TransportMLP(d_in, d_out).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-5
    )

    # Normalize inputs for stable training
    src_mean = source_train.mean(dim=0, keepdim=True)
    src_std = source_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    tgt_mean = target_train.mean(dim=0, keepdim=True)
    tgt_std = target_train.std(dim=0, keepdim=True).clamp(min=1e-6)

    src_train_n = (source_train - src_mean) / src_std
    tgt_train_n = (target_train - tgt_mean) / tgt_std
    src_test_n = (source_test - src_mean) / src_std

    # Training loop with cosine annealing
    model.train()
    best_loss = float("inf")
    for step in range(n_steps):
        pred = model(src_train_n)
        loss = nn.functional.mse_loss(pred, tgt_train_n)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()  # type: ignore[no-untyped-call]
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val

        if step % 500 == 0:
            logger.debug(
                "MLP transport step %d/%d, loss=%.6f, lr=%.2e",
                step, n_steps, loss_val, scheduler.get_last_lr()[0],
            )

    final_loss = loss_val  # type: ignore[possibly-undefined]
    converged = final_loss < 2.0 * best_loss + 1e-8

    # Evaluate with per-dimension R²
    model.eval()
    with torch.no_grad():
        pred_test_n = model(src_test_n)
        # Denormalize
        pred_test = pred_test_n * tgt_std + tgt_mean

        ss_res_d = ((target_test - pred_test) ** 2).sum(dim=0)
        ss_tot_d = ((target_test - target_test.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)

        if ss_tot_d.max() < 1e-10:
            return 1.0, converged
        r2_d = 1.0 - ss_res_d / ss_tot_d.clamp(min=1e-10)
        r2 = r2_d.clamp(min=0.0).mean().item()
        return r2, converged


def transport_linearity_score(
    features_a: Tensor,
    features_b: Tensor,
    train_ratio: float = 0.8,
    mlp_steps: int = 3000,
    mlp_lr: float = 1e-3,
    seed: int = 42,
) -> TransportResult:
    """Compute Transport Linearity Score τ(A, B).

    τ = Perf(M_linear) / Perf(M_best) where Perf is R² of reconstruction.
    Measures whether a linear map suffices for cross-model transport.

    Args:
        features_a: Source feature matrix, shape (N, D_a).
        features_b: Target feature matrix, shape (N, D_b).
        train_ratio: Fraction of data used for fitting (rest for evaluation).
        mlp_steps: Number of Adam steps for MLP training.
        mlp_lr: Learning rate for MLP.
        seed: Random seed for reproducibility.

    Returns:
        TransportResult with τ, linear performance, and best performance.

    Raises:
        ValueError: If feature matrices have different number of samples.
    """
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            f"Feature matrices must have same N: "
            f"got {features_a.shape[0]} and {features_b.shape[0]}"
        )
    n = features_a.shape[0]
    n_train = int(n * train_ratio)

    # Deterministic split
    torch.manual_seed(seed)  # type: ignore[arg-type]
    perm = torch.randperm(n, device=features_a.device)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    src_train, src_test = features_a[train_idx], features_a[test_idx]
    tgt_train, tgt_test = features_b[train_idx], features_b[test_idx]

    # Linear transport (Procrustes)
    perf_linear, w_linear = _evaluate_linear_transport(src_train, tgt_train, src_test, tgt_test)

    # Best transport (MLP)
    perf_best, mlp_converged = _evaluate_mlp_transport(
        src_train,
        tgt_train,
        src_test,
        tgt_test,
        n_steps=mlp_steps,
        lr=mlp_lr,
        seed=seed,
    )

    # τ = perf_linear / perf_best, clamped to [0, 1]
    # When both transports fail (R² ≈ 0), tau=0: no evidence linear works
    tau = 0.0 if perf_best < 1e-10 else min(1.0, max(0.0, perf_linear / perf_best))

    if perf_linear > perf_best + 0.02:
        logger.warning(
            "perf_linear (%.4f) > perf_best (%.4f): MLP may have underfit, "
            "resulting in tau=min(1.0, ratio)=%.4f. Consider increasing mlp_steps.",
            perf_linear, perf_best, tau,
        )

    return TransportResult(
        perf_linear=perf_linear,
        perf_best=perf_best,
        tau=tau,
        map_linear_weights=w_linear,
        converged=mlp_converged,
    )


def transport_asymmetry_index(
    features_a: Tensor,
    features_b: Tensor,
    train_ratio: float = 0.8,
    mlp_steps: int = 3000,
    mlp_lr: float = 1e-3,
    seed: int = 42,
) -> BidirectionalTransportResult:
    """Compute Transport Asymmetry Index TAI(A, B).

    TAI = |Perf(A→B) - Perf(B→A)| / max(Perf(A→B), Perf(B→A))
    Measures directional imbalance in cross-model transport.

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).
        train_ratio: Fraction of data for fitting.
        mlp_steps: Number of Adam steps for MLP.
        mlp_lr: Learning rate for MLP.
        seed: Random seed.

    Returns:
        BidirectionalTransportResult with TAI and both directional results.
    """
    a_to_b = transport_linearity_score(
        features_a,
        features_b,
        train_ratio=train_ratio,
        mlp_steps=mlp_steps,
        mlp_lr=mlp_lr,
        seed=seed,
    )
    b_to_a = transport_linearity_score(
        features_b,
        features_a,
        train_ratio=train_ratio,
        mlp_steps=mlp_steps,
        mlp_lr=mlp_lr,
        seed=seed,
    )

    # TAI uses the best (MLP) performance in each direction
    perf_ab = a_to_b.perf_best
    perf_ba = b_to_a.perf_best
    max_perf = max(perf_ab, perf_ba)

    tai = 0.0 if max_perf < 1e-10 else abs(perf_ab - perf_ba) / max_perf

    direction = "A→B" if perf_ab >= perf_ba else "B→A"

    return BidirectionalTransportResult(
        a_to_b=a_to_b,
        b_to_a=b_to_a,
        tai=tai,
        asymmetry_direction=direction,
    )
