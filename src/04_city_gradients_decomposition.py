"""City-level income gradients, Holm multiplicity control, and 3-component route decomposition.

Implements Methods ("Adjusted route responses and city income gradients", "Decomposing
city responses") and Supplementary Methods S3 & S4:
  - Forward route-composition identity (exact additive decomposition):
      100 ln(E_H / E_R) = 100 ln(T_H / T_R)
                          + 100 ln[(C / T_H) / (E_R / T_R)]
                          + 100 ln(E_H / C),
    where C = sum_r t_{H,r} * (e_{R,r} / t_{R,r}).
  - Reverse-order sensitivity on positive-heat-period route support using
    B = sum_r t_{R,r} * (e_{H,r} / t_{H,r}) and arithmetic-average allocation.
  - Province-clustered CR1 WLS (routine emissions weights) and OLS (equal-city weights)
    regressions with aligned province influence vectors for paired contrasts and Holm
    family-wise multiplicity adjustment.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def decompose_city_route_response(
    t_routine: np.ndarray,
    t_heat: np.ndarray,
    e_routine: np.ndarray,
    e_heat: np.ndarray,
    include_reverse: bool = False,
) -> Dict[str, float]:
    """Decompose a single city's emissions response into travel intensity, route mix, and within-route.

    Parameters
    ----------
    t_routine, t_heat : 1-D arrays
        Route-level conditional mean travel intensity on routine and heatwave days.
    e_routine, e_heat : 1-D arrays
        Route-level conditional mean emissions on routine and heatwave days.
    include_reverse : bool, default False
        If True, also compute reverse-order and arithmetic-average allocations (requires
        strictly positive `t_heat` on every included route; no zero imputation).
    """
    tr = np.asarray(t_routine, dtype=np.float64)
    th = np.asarray(t_heat, dtype=np.float64)
    er = np.asarray(e_routine, dtype=np.float64)
    eh = np.asarray(e_heat, dtype=np.float64)

    if any(a.shape != tr.shape or a.ndim != 1 for a in (tr, th, er, eh)) or len(tr) == 0:
        raise ValueError("Non-empty, matching 1-D route vectors are required.")
    if not all(np.isfinite(a).all() for a in (tr, th, er, eh)):
        raise ValueError("All route inputs must be finite.")
    if (tr <= 0.0).any() or (er <= 0.0).any() or (th < 0.0).any() or (eh < 0.0).any():
        raise ValueError("Routine travel/emissions must be positive; heat-period inputs nonnegative.")
    if ((th == 0.0) & (eh != 0.0)).any():
        raise ValueError("Zero heat-period travel with positive emissions is inconsistent.")

    T_R, T_H = float(tr.sum()), float(th.sum())
    E_R, E_H = float(er.sum()), float(eh.sum())
    if min(T_R, T_H, E_R, E_H) <= 0.0:
        raise ValueError("City totals T_R, T_H, E_R, and E_H must be strictly positive.")

    C = float(np.dot(th, er / tr))
    I_R, I_H = E_R / T_R, E_H / T_H

    out = {
        "total": float(100.0 * np.log(E_H / E_R)),
        "travel_intensity": float(100.0 * np.log(T_H / T_R)),
        "route_mix_forward": float(100.0 * np.log((C / T_H) / I_R)),
        "within_route_forward": float(100.0 * np.log(E_H / C)),
        "C": C,
    }

    if include_reverse:
        if (th <= 0.0).any():
            raise ValueError("Reverse decomposition requires strictly positive heat-period route travel.")
        B = float(np.dot(tr, eh / th))
        out["B"] = B
        out["within_route_reverse"] = float(100.0 * np.log(B / E_R))
        out["route_mix_reverse"] = float(100.0 * np.log(I_H / (B / T_R)))
        out["route_mix_average"] = 0.5 * (out["route_mix_forward"] + out["route_mix_reverse"])
        out["within_route_average"] = 0.5 * (out["within_route_forward"] + out["within_route_reverse"])

    return out


def fit_province_clustered_multi_outcome(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    provinces: np.ndarray,
    income_col_idx: int,
    min_clusters: int = 20,
) -> Dict[str, object]:
    """Fit multi-outcome WLS/OLS city models on identical samples and return province influences.

    Because every component regression shares the exact design matrix X, city sample, and
    weights, the income coefficients and province-level CR1 influence vectors are exactly
    additive across total = travel_intensity + route_mix + within_route.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    groups = np.asarray(provinces)

    n, k = X.shape
    if Y.ndim != 2 or Y.shape[0] != n or len(w) != n or len(groups) != n:
        raise ValueError("Input dimensions do not align.")
    if (w <= 0.0).any() or not (np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(w).all()):
        raise ValueError("Finite inputs and strictly positive weights are required.")
    if np.linalg.matrix_rank(X) != k:
        raise ValueError("Design matrix X is rank-deficient.")

    prov_ids, prov_idx = np.unique(groups, return_inverse=True)
    G = len(prov_ids)
    if G < min_clusters:
        raise ValueError(f"At least {min_clusters} province clusters required; found {G}.")

    W = w / w.mean()
    bread = np.linalg.inv(X.T @ (W[:, None] * X))
    B = bread @ (X.T @ (W[:, None] * Y))
    residuals = Y - X @ B

    # CR1 finite-sample scaled province influence vectors psi_{g, m}
    row_proj = (X @ bread[:, income_col_idx]) * W
    psi = np.zeros((G, Y.shape[1]), dtype=np.float64)
    np.add.at(psi, prov_idx, residuals * row_proj[:, None])
    cr1_factor = np.sqrt((G / (G - 1.0)) * ((n - 1.0) / (n - k)))
    psi *= cr1_factor

    cov = psi.T @ psi
    coefs = B[income_col_idx]
    ses = np.sqrt(np.maximum(0.0, np.diag(cov)))
    df = G - 1
    crit = float(stats.t.ppf(0.975, df))
    p_vals = 2.0 * stats.t.sf(np.abs(coefs / np.where(ses > 0, ses, np.nan)), df)

    return {
        "coefs": coefs,
        "ses": ses,
        "ci_low": coefs - crit * ses,
        "ci_high": coefs + crit * ses,
        "p_values": p_vals,
        "cov": cov,
        "influences": psi,
        "province_ids": prov_ids,
        "n_cities": n,
        "n_provinces": G,
        "df": df,
    }


def apply_holm_adjustment(p_values: Sequence[float]) -> np.ndarray:
    """Apply Holm-Bonferroni multiplicity correction within a predefined test family."""
    _, p_adj, _, _ = multipletests(np.asarray(p_values, dtype=np.float64), method="holm")
    return p_adj
