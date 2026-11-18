"""Alignment metrics: mutual k-NN, CKA, and Alignment Locality (λ).

Implements §2.1, §2.3 of the BCCT framework. All functions operate on
pre-computed representation matrices (N, D) as torch Tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _pairwise_distances(x: Tensor, y: Tensor | None = None) -> Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
        x: Feature matrix, shape (N, D).
        y: Optional second feature matrix, shape (N, D). If None, computes self-distances.

    Returns:
        Distance matrix, shape (N, N).
    """
    if y is None:
        y = x
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    x_sq = (x * x).sum(dim=1, keepdim=True)  # (N, 1)
    y_sq = (y * y).sum(dim=1, keepdim=True)  # (N, 1)
    dist = x_sq + y_sq.T - 2.0 * x @ y.T
    return dist.clamp(min=0.0)


def _knn_indices(x: Tensor, k: int) -> Tensor:
    """Compute k-nearest neighbor indices for each point.

    Args:
        x: Feature matrix, shape (N, D).
        k: Number of nearest neighbors.

    Returns:
        Neighbor indices, shape (N, k). Excludes self.
    """
    dist = _pairwise_distances(x)
    # Set self-distance to inf to exclude self from neighbors
    dist.fill_diagonal_(float("inf"))
    # topk with largest=False gives the k smallest distances
    _, indices = dist.topk(k, dim=1, largest=False)
    return indices


def mutual_knn_overlap(
    features_a: Tensor,
    features_b: Tensor,
    k: int = 10,
) -> float:
    """Compute mutual k-nearest neighbor overlap between two feature sets.

    Measures the fraction of shared k-NN sets between two representations,
    as defined in Huh et al. (2024), "The Platonic Representation Hypothesis".

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).
        k: Number of nearest neighbors to consider.

    Returns:
        Raw overlap score in [0, 1], where 1 means identical neighborhoods.

    Raises:
        ValueError: If feature matrices have different number of samples.
    """
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            f"Feature matrices must have same N: "
            f"got {features_a.shape[0]} and {features_b.shape[0]}"
        )
    n = features_a.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than N={n}")

    nn_a = _knn_indices(features_a, k)  # (N, k)
    nn_b = _knn_indices(features_b, k)  # (N, k)

    # Convert to sets and compute overlap per point
    overlap_sum = 0.0
    for i in range(n):
        set_a: set[int] = set(nn_a[i].tolist())  # type: ignore[arg-type]
        set_b: set[int] = set(nn_b[i].tolist())  # type: ignore[arg-type]
        overlap_sum += len(set_a & set_b) / k

    return overlap_sum / n


@torch.no_grad()
def mutual_knn_overlap_fast(
    features_a: Tensor,
    features_b: Tensor,
    k: int = 10,
) -> float:
    """Vectorized mutual k-NN overlap — faster for large N.

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).
        k: Number of nearest neighbors.

    Returns:
        Raw overlap score in [0, 1].
    """
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            f"Feature matrices must have same N: "
            f"got {features_a.shape[0]} and {features_b.shape[0]}"
        )
    n = features_a.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than N={n}")

    nn_a = _knn_indices(features_a, k)  # (N, k)
    nn_b = _knn_indices(features_b, k)  # (N, k)

    # Build binary membership matrices (N, N) — nn_a_mask[i, j] = 1 if j in kNN(i) in A
    nn_a_mask = torch.zeros(n, n, dtype=torch.bool, device=features_a.device)
    nn_b_mask = torch.zeros(n, n, dtype=torch.bool, device=features_b.device)

    row_idx = torch.arange(n, device=features_a.device).unsqueeze(1).expand_as(nn_a)
    nn_a_mask[row_idx, nn_a] = True
    nn_b_mask[row_idx, nn_b] = True

    # Overlap = number of shared neighbors per point
    overlap = (nn_a_mask & nn_b_mask).sum(dim=1).float()  # (N,)
    return (overlap / k).mean().item()


@torch.no_grad()
def null_calibrated_mutual_knn(
    features_a: Tensor,
    features_b: Tensor,
    k: int = 10,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Null-calibrated mutual k-NN overlap (Aristotelian refinement).

    Calibrates the raw mutual k-NN overlap against a permutation-based null
    distribution, following Gröger et al. (2026). Returns a calibrated score
    in [0, 1] where 0 means chance-level alignment and 1 means perfect alignment.

    Calibrated score = (raw - null_mean) / (1 - null_mean)

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).
        k: Number of nearest neighbors.
        n_permutations: Number of random permutations for null distribution.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (calibrated_overlap, raw_overlap, null_mean):
            - calibrated_overlap: S_local in [0, 1], null-calibrated alignment.
            - raw_overlap: The raw mutual k-NN overlap.
            - null_mean: Mean of the null distribution.
    """
    raw = mutual_knn_overlap_fast(features_a, features_b, k)

    generator = torch.Generator(device=features_b.device)
    generator.manual_seed(seed)

    null_scores: list[float] = []
    n = features_b.shape[0]

    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=generator, device=features_b.device)
        features_b_perm = features_b[perm]
        null_score = mutual_knn_overlap_fast(features_a, features_b_perm, k)
        null_scores.append(null_score)

    null_tensor = torch.tensor(null_scores)
    null_mean = null_tensor.mean().item()

    # Calibrated overlap: excess above chance, normalized to [0, 1]
    if (1.0 - null_mean) < 1e-10:
        calibrated = 0.0 if abs(raw - null_mean) < 1e-10 else 1.0
    else:
        calibrated = max(0.0, (raw - null_mean) / (1.0 - null_mean))

    return calibrated, raw, null_mean


@torch.no_grad()
def centered_kernel_alignment(
    features_a: Tensor,
    features_b: Tensor,
) -> float:
    """Compute linear Centered Kernel Alignment (CKA).

    CKA measures global representational similarity as the RV coefficient
    between centered kernel matrices, following Kornblith et al. (2019).

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).

    Returns:
        CKA score in [0, 1], where 1 means identical kernel structure.

    Raises:
        ValueError: If feature matrices have different number of samples.
    """
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            f"Feature matrices must have same N: "
            f"got {features_a.shape[0]} and {features_b.shape[0]}"
        )

    # Center the features (subtract column means)
    a = features_a - features_a.mean(dim=0, keepdim=True)
    b = features_b - features_b.mean(dim=0, keepdim=True)

    # Compute linear kernels
    # HSIC(K, L) = trace(KHLH) / (n-1)^2 where H = I - 11'/n
    # For centered features: HSIC = ||A^T B||_F^2 / (n-1)^2
    cross = float(torch.linalg.norm(a.T @ b, ord="fro").item() ** 2)  # type: ignore[reportUnknownMemberType]
    norm_a = float(torch.linalg.norm(a.T @ a, ord="fro").item() ** 2)  # type: ignore[reportUnknownMemberType]
    norm_b = float(torch.linalg.norm(b.T @ b, ord="fro").item() ** 2)  # type: ignore[reportUnknownMemberType]

    denom = (norm_a * norm_b) ** 0.5
    if denom < 1e-10:
        return 0.0

    return cross / denom


@torch.no_grad()
def alignment_locality(
    features_a: Tensor,
    features_b: Tensor,
    k: int = 10,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute Alignment Locality λ = S_local / S_global.

    Measures the ratio of local (neighborhood) alignment to global (kernel)
    alignment, distinguishing Aristotelian from Platonic convergence.

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).
        k: Number of nearest neighbors for local alignment.
        n_permutations: Number of permutations for null calibration.
        seed: Random seed.

    Returns:
        Tuple of (lambda_value, s_local_calibrated, s_global_cka):
            - lambda_value: λ = S_local / S_global.
            - s_local_calibrated: Null-calibrated mutual k-NN overlap.
            - s_global_cka: Linear CKA score.
    """
    s_local, _, _ = null_calibrated_mutual_knn(
        features_a, features_b, k=k, n_permutations=n_permutations, seed=seed
    )
    s_global = centered_kernel_alignment(features_a, features_b)

    lambda_val = (float("inf") if s_local > 0 else 0.0) if s_global < 1e-10 else s_local / s_global

    return lambda_val, s_local, s_global
