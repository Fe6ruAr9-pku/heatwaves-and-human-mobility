"""2050 SSP5-8.5 cooling, city-tier electrification, and crossed flow-calendar accounting.

Implements Methods ("Future and policy scenarios") and Supplementary Methods S6:
  - Ambient cooling scenarios (-0.5, -1.0, -1.5, -2.0 deg C) evaluated across the full
    2 x 2 matrix of modeled flows (0 vs. c) and reconstructed 35 deg C heatwave calendars
    (0 vs. c): M_00, M_c0, M_0c, and M_cc.
  - Exact symmetric flow-vs-calendar decomposition:
      flow_sym  = [(M_00 - M_c0) + (M_0c - M_cc)] / 2
      label_sym = [(M_00 - M_0c) + (M_c0 - M_cc)] / 2
      flow_sym + label_sym = M_00 - M_cc.
  - City-tier passenger-car electrification transferring fraction `a` (0.8 or 1.0) of
    existing ICE car share to BEV share at targeted origins while leaving other NEV shares
    unchanged:
      theta'_ICE = (1 - a) * theta_ICE
      theta'_BEV = theta_BEV + a * theta_ICE.
  - Direct recomputation of joint cooling + electrification scenarios and exact reporting
    of their non-additive interaction (joint - cooling_alone - ev_alone).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from . import __name__ as _pkg  # noqa: F401


EMISSION_FACTORS_G_PKM: Dict[str, tuple[float, float, float, float, float]] = {
    "main": (135.0, 55.0, 77.0, 19.0, 123.0),
    "high_labeled": (163.0, 125.0, 165.0, 101.0, 98.0),
    "low_labeled": (101.0, 0.0, 32.0, 6.0, 133.0),
}


def transfer_origin_fleet_to_bev(
    ice_share: np.ndarray,
    bev_share: np.ndarray,
    other_nev_share: np.ndarray,
    origin_tiers: Sequence[str],
    target_tiers: Sequence[str],
    transfer_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transfer fraction `a` of origin ICE car share to BEV share in targeted income tiers."""
    if not (0.0 <= transfer_fraction <= 1.0):
        raise ValueError("transfer_fraction must lie in [0, 1].")
    ice = np.asarray(ice_share, dtype=np.float64)
    bev = np.asarray(bev_share, dtype=np.float64)
    onev = np.asarray(other_nev_share, dtype=np.float64)
    mask = np.isin(np.asarray(origin_tiers), list(target_tiers))

    transferred = np.where(mask, transfer_fraction * ice, 0.0)
    ice_new = ice - transferred
    bev_new = bev + transferred
    return ice_new, bev_new, onev.copy()


def route_metrics_from_pkm(
    hot_pkm: np.ndarray,
    routine_pkm: np.ndarray,
    n_hot: np.ndarray,
    n_routine: np.ndarray,
    ice_share: np.ndarray,
    bev_share: np.ndarray,
    other_nev_share: np.ndarray,
    factor_set: str = "main",
) -> Dict[str, float]:
    """Compute common-route G (heatwave-coinciding), D (excess), and T (all-day) totals (tonnes).

    Parameters
    ----------
    hot_pkm, routine_pkm : np.ndarray, shape (n_routes, 3)
        Aggregated passenger-kilometers for (car, rail, air) on heatwave and routine days.
    n_hot, n_routine : np.ndarray, shape (n_routes,)
        Count of heatwave and routine records per route. Routes with n_hot == 0 remain in
        the support with G = D = 0.
    """
    f_ice, f_bev, f_onev, f_rail, f_air = EMISSION_FACTORS_G_PKM[factor_set]
    car_ef = ice_share * f_ice + bev_share * f_bev + other_nev_share * f_onev
    ef_t_per_pkm = np.column_stack(
        [car_ef, np.full(len(car_ef), f_rail), np.full(len(car_ef), f_air)]
    ) * 1e-6

    h_t = hot_pkm * ef_t_per_pkm
    r_t = routine_pkm * ef_t_per_pkm

    nh = np.asarray(n_hot, dtype=np.float64)[:, None]
    nr = np.asarray(n_routine, dtype=np.float64)[:, None]
    if (nr <= 0).any():
        raise ValueError("Every common route must have at least one routine record.")

    baseline_ref = nh * (r_t / nr)
    gap_t = h_t - baseline_ref
    all_t = h_t + r_t

    return {
        "G_tonnes": float(h_t.sum()),
        "D_tonnes": float(gap_t.sum()),
        "T_tonnes": float(all_t.sum()),
    }


def decompose_cooling_flow_calendar(
    m00: float,
    mc0: float,
    m0c: float,
    mcc: float,
) -> Dict[str, float]:
    """Decompose total cooling reduction (M_00 - M_cc) into flow and calendar contributions.

    Matches Supplementary Methods S6:
      - Within-episode flow only (fixed baseline calendar): M_00 - M_c0
      - Total cooling effect (recomputed 35 deg C calendar): M_00 - M_cc
      - Symmetric flow contribution:     [(M_00 - M_c0) + (M_0c - M_cc)] / 2
      - Symmetric calendar contribution: [(M_00 - M_0c) + (M_c0 - M_cc)] / 2
    """
    total_red = m00 - mcc
    flow_first = m00 - mc0
    label_after_flow = mc0 - mcc
    flow_sym = 0.5 * ((m00 - mc0) + (m0c - mcc))
    label_sym = 0.5 * ((m00 - m0c) + (mc0 - mcc))
    return {
        "total_reduction": float(total_red),
        "within_episode_flow_only": float(flow_first),
        "avoided_heatwave_days_after_flow": float(label_after_flow),
        "flow_symmetric": float(flow_sym),
        "calendar_symmetric": float(label_sym),
    }


def evaluate_joint_interaction(
    baseline: float,
    cooling_remaining: float,
    ev_remaining: float,
    joint_remaining: float,
) -> Dict[str, float]:
    """Compute standalone, joint, and interaction reductions relative to a shared baseline."""
    cool_red = baseline - cooling_remaining
    ev_red = baseline - ev_remaining
    joint_red = baseline - joint_remaining
    return {
        "baseline": float(baseline),
        "cooling_reduction": float(cool_red),
        "ev_reduction": float(ev_red),
        "joint_reduction": float(joint_red),
        "interaction": float(joint_red - cool_red - ev_red),
        "cooling_pct": float(100.0 * cool_red / baseline) if baseline > 0 else np.nan,
        "ev_pct": float(100.0 * ev_red / baseline) if baseline > 0 else np.nan,
        "joint_pct": float(100.0 * joint_red / baseline) if baseline > 0 else np.nan,
    }
