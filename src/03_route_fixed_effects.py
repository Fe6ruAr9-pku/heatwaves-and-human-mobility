"""Memory-bounded weighted fixed-effects estimation and cross-year origin-clustered inference.

Implements the route-level econometric estimator in Methods ("Adjusted route responses and
city income gradients") and Supplementary Methods S3 & S5:
  - Model: ln(TI_{odt}) = alpha_{od,y} + lambda_t + gamma_{o,y,w(t)}
                          + sum_k beta_{yk} * HW_{ot} * I(g_d = k) + eps_{odt}
  - Weighted alternating projections (Frisch-Waugh-Lovell) absorb directed route, calendar
    date, and origin-by-ISO-week fixed effects in double precision (tol = 1e-8; post-fit
    weighted group-mean orthogonality checked against 5e-8).
  - Origin-clustered CR1 sandwich covariance aligns cluster scores across years to retain
    cross-year covariance for early (2019-2021) vs. late (2022-2024) period contrasts,
    high-minus-low destination gaps, and leave-one-year-out checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import linalg, stats

BLOCK_SIZE = 500_000


def _projection_weight_totals(
    fe_codes: Sequence[np.ndarray],
    weights: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> List[np.ndarray]:
    """Compute weighted observation counts per absorbed fixed-effect category."""
    totals: List[np.ndarray] = []
    n = len(weights)
    for codes in fe_codes:
        k = int(codes.max()) + 1
        den = np.zeros(k, dtype=np.float64)
        for start in range(0, n, block_size):
            sl = slice(start, min(n, start + block_size))
            den += np.bincount(codes[sl], weights=weights[sl], minlength=k)
        totals.append(den)
    return totals


def residualize_weighted_fe(
    z: np.ndarray,
    fe_codes: Sequence[np.ndarray],
    weights: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 500,
    block_size: int = BLOCK_SIZE,
) -> Dict[str, float]:
    """In-place double-precision weighted alternating projections across arbitrary FE dimensions.

    Parameters
    ----------
    z : np.ndarray, shape (N, 1 + p)
        Stacked outcome and regressor matrix modified in place.
    fe_codes : sequence of 1-D integer arrays
        Zero-based factorized fixed-effect group codes (e.g., [route, date, origin_week]).
    weights : np.ndarray, shape (N,)
        Positive observation weights (annual route routine-emissions means, normalized).
    tol : float, default 1e-8
        Convergence tolerance on maximum absolute weighted group mean update.

    Returns
    -------
    dict
        Convergence diagnostics including iterations, final step change, and max residual
        weighted group mean (`orthogonality`).
    """
    n, ncols = z.shape
    totals = _projection_weight_totals(fe_codes, weights, block_size=block_size)
    sizes = [len(t) for t in totals]

    for iteration in range(1, max_iter + 1):
        max_change = 0.0
        for codes, k, den in zip(fe_codes, sizes, totals):
            sums = np.zeros((k, ncols), dtype=np.float64)
            for start in range(0, n, block_size):
                sl = slice(start, min(n, start + block_size))
                for j in range(ncols):
                    sums[:, j] += np.bincount(
                        codes[sl], weights=z[sl, j] * weights[sl], minlength=k
                    )
            means = np.divide(sums, den[:, None], out=np.zeros_like(sums), where=den[:, None] > 0)
            max_change = max(max_change, float(np.abs(means).max()))
            for start in range(0, n, block_size):
                sl = slice(start, min(n, start + block_size))
                z[sl] -= means[codes[sl]]
        if max_change < tol:
            orth = max_weighted_group_mean(z, fe_codes, weights, totals, block_size=block_size)
            return {"iterations": iteration, "change": max_change, "orthogonality": orth}

    raise RuntimeError(f"Alternating projections did not converge within {max_iter} iterations.")


def max_weighted_group_mean(
    z: np.ndarray,
    fe_codes: Sequence[np.ndarray],
    weights: np.ndarray,
    totals: Sequence[np.ndarray],
    block_size: int = BLOCK_SIZE,
) -> float:
    """Verify post-projection orthogonality against all absorbed fixed effects."""
    n, ncols = z.shape
    worst = 0.0
    for codes, den in zip(fe_codes, totals):
        k = len(den)
        for j in range(ncols):
            num = np.zeros(k, dtype=np.float64)
            for start in range(0, n, block_size):
                sl = slice(start, min(n, start + block_size))
                num += np.bincount(codes[sl], weights=weights[sl] * z[sl, j], minlength=k)
            means = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            worst = max(worst, float(np.abs(means).max()))
    return worst


@dataclass
class AnnualRouteFit:
    """Container for one annual fixed-effects route regression."""

    year: int
    beta: np.ndarray
    hessian: np.ndarray
    origin_scores: np.ndarray
    active_origins: np.ndarray
    n_obs: int
    fe_group_counts: tuple[int, ...]
    orthogonality: float


def fit_annual_route_fe(
    year: int,
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    fe_codes: Sequence[np.ndarray],
    origin_ids: np.ndarray,
    n_total_origins: int,
    tol: float = 1e-8,
    orth_limit: float = 5e-8,
    block_size: int = BLOCK_SIZE,
) -> AnnualRouteFit:
    """Fit one annual route-level fixed-effects model and return cluster-aligned scores."""
    n, p = x.shape
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.mean()

    z = np.empty((n, 1 + p), dtype=np.float64)
    z[:, 0] = y
    z[:, 1:] = x

    conv = residualize_weighted_fe(z, fe_codes, w, tol=tol, block_size=block_size)
    if conv["orthogonality"] > orth_limit:
        raise RuntimeError(
            f"Fixed-effect orthogonality check failed ({conv['orthogonality']:.2e} > {orth_limit:.2e})."
        )

    y_res = z[:, 0]
    x_res = z[:, 1:]

    hess = np.zeros((p, p), dtype=np.float64)
    rhs = np.zeros(p, dtype=np.float64)
    for start in range(0, n, block_size):
        sl = slice(start, min(n, start + block_size))
        hess += x_res[sl].T @ (w[sl, None] * x_res[sl])
        rhs += x_res[sl].T @ (w[sl] * y_res[sl])
    beta = np.linalg.solve(hess, rhs)

    # Aggregate weighted cluster scores S_g by origin city for cross-year alignment
    scores = np.zeros((n_total_origins, p), dtype=np.float64)
    for start in range(0, n, block_size):
        sl = slice(start, min(n, start + block_size))
        u = y_res[sl] - x_res[sl] @ beta
        for j in range(p):
            scores[:, j] += np.bincount(
                origin_ids[sl], weights=x_res[sl, j] * w[sl] * u, minlength=n_total_origins
            )

    active = np.zeros(n_total_origins, dtype=bool)
    active[np.unique(origin_ids)] = True
    fe_counts = tuple(int(c.max()) + 1 for c in fe_codes)

    return AnnualRouteFit(
        year=year,
        beta=beta,
        hessian=hess,
        origin_scores=scores,
        active_origins=active,
        n_obs=n,
        fe_group_counts=fe_counts,
        orthogonality=conv["orthogonality"],
    )


def stacked_cr1_covariance(
    fits: Sequence[AnnualRouteFit],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Form stacked cross-year coefficient vector, CR0 covariance V0, and CR1 covariance V1.

    Matches Supplementary Methods S3:
      V_0 = A^{-1} (sum_g S_g S_g') A^{-1}
      V_1 = [G / (G - 1)] * [(N - 1) / (N - p - K_FE)] * V_0
    where K_FE equals the sum of absorbed group counts across years minus two.
    """
    beta = np.concatenate([f.beta for f in fits])
    a_mat = linalg.block_diag(*[f.hessian for f in fits])
    a_inv = np.linalg.inv(a_mat)
    stacked_scores = np.column_stack([f.origin_scores for f in fits])

    n_total = sum(f.n_obs for f in fits)
    k_fe = sum(sum(f.fe_group_counts) for f in fits) - 2
    g_clusters = int(np.logical_or.reduce([f.active_origins for f in fits]).sum())

    v0 = a_inv @ (stacked_scores.T @ stacked_scores) @ a_inv
    cr1_scale = (g_clusters / (g_clusters - 1.0)) * (
        (n_total - 1.0) / (n_total - len(beta) - k_fe)
    )
    v1 = cr1_scale * v0
    return beta, v0, v1, g_clusters


def linear_contrast(
    beta: np.ndarray,
    cov: np.ndarray,
    c: np.ndarray,
    df: int,
    scale: float = 100.0,
) -> Dict[str, float]:
    """Evaluate a linear combination c'beta (in 100 natural-log points) with Student-t inference."""
    est = float(c @ beta) * scale
    se = float(np.sqrt(max(0.0, float(c @ cov @ c)))) * scale
    crit = float(stats.t.ppf(0.975, df))
    p_val = float(2.0 * stats.t.sf(abs(est / se), df)) if se > 0.0 else 1.0
    return {
        "estimate": est,
        "se": se,
        "ci_low": est - crit * se,
        "ci_high": est + crit * se,
        "p_value": p_val,
        "df": df,
    }


def build_period_contrast_vector(
    years: Sequence[int],
    n_terms_per_year: int,
    early_end: int = 2021,
    high_idx: int = 2,
    low_idx: int | None = None,
) -> np.ndarray:
    """Build equal-year contrast vector comparing late (> early_end) minus early (<= early_end).

    Set `low_idx = 0` to evaluate the high-minus-low destination gap contrast.
    """
    early = [y for y in years if y <= early_end]
    late = [y for y in years if y > early_end]
    c = np.zeros(len(years) * n_terms_per_year, dtype=np.float64)
    for i, yr in enumerate(years):
        w = (1.0 / len(late)) if yr in late else (-1.0 / len(early))
        c[i * n_terms_per_year + high_idx] = w
        if low_idx is not None:
            c[i * n_terms_per_year + low_idx] = -w
    return c
