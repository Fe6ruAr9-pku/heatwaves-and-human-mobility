"""Harmonized route-level emissions accounting, conditional city indices, and mode-model specs.

Implements the accounting framework in Methods ("Mobility allocation and emissions
accounting") and Supplementary Methods S2 & Supplementary Table S2:
  - Fixed well-to-wheel CO2-equivalent emission factors (g CO2-eq per passenger-km) for
    ICE cars, BEVs, other NEVs, rail, and aviation across Main, High-labeled, and
    Low-labeled factor combinations.
  - Origin-specific fleet-weighted car factor and harmonized air-share cutoff (< 0.05 set
    to zero without redistribution).
  - Route-date emissions: e_{rt} = 10^{-6} * d_r * sum_m (p_{rtm} * f_{om}).
  - Annual common-route totals G_y (heatwave-coinciding), D_y (heat-period excess over
    same-route routine mean), and T_y (all-record emissions), plus US EPA (2023) social
    cost of carbon valuation ($190 per tonne CO2).
  - Historical conditional city indices H = sum_r e_bar_{H,r}, R = sum_r e_bar_{R,r}, and
    log-ratio emissions response 100 * ln(H / R).
  - Reference XGBoost regressor configuration for historical mode shares and future flows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

# Well-to-wheel emission factor combinations (g CO2-eq per passenger-kilometer)
# Keys: (ice_car, bev_car, other_nev_car, rail, air) — Supplementary Table S2
EMISSION_FACTOR_SETS: Dict[str, Dict[str, float]] = {
    "main": {"ice": 135.0, "bev": 55.0, "other_nev": 77.0, "rail": 19.0, "air": 123.0},
    "high_labeled": {"ice": 163.0, "bev": 125.0, "other_nev": 165.0, "rail": 101.0, "air": 98.0},
    "low_labeled": {"ice": 101.0, "bev": 0.0, "other_nev": 32.0, "rail": 6.0, "air": 133.0},
}

# US EPA (2023) central social cost of carbon (USD per tonne CO2)
EPA_CENTRAL_SCC_USD_PER_TONNE: float = 190.0

# Sixteen predictors used by the historical mode-share XGBoost models (Supplementary Table S9)
MODE_SHARE_PREDICTORS = (
    "temax",
    "prec",
    "SNDP",
    "WDSP",
    "VISIB",
    "calendar",
    "distance",
    "GDP",
    "population",
    "NDVI",
    "RoadFV",
    "RoadPV",
    "RailFV",
    "RailPV",
    "AirFV",
    "AirPV",
)


@dataclass(frozen=True)
class XGBoostModelSpec:
    """Hyperparameter defaults for historical mode-share and future mode-flow regressors."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "reg:squarederror"
    random_state: int = 42


HISTORICAL_SHARE_SPEC = XGBoostModelSpec(max_depth=6)
FUTURE_FLOW_SPEC = XGBoostModelSpec(max_depth=8)


def fleet_weighted_car_factor(
    ice_share: np.ndarray,
    bev_share: np.ndarray,
    other_nev_share: np.ndarray,
    factor_set: str = "main",
) -> np.ndarray:
    """Compute origin-specific fleet-weighted passenger-car factor (g CO2-eq / pkm)."""
    f = EMISSION_FACTOR_SETS[factor_set]
    return ice_share * f["ice"] + bev_share * f["bev"] + other_nev_share * f["other_nev"]


def apply_harmonized_air_cutoff(
    air_share: np.ndarray,
    cutoff: float = 0.05,
) -> np.ndarray:
    """Zero out modeled air shares below `cutoff` without redistributing removed share."""
    arr = np.asarray(air_share, dtype=float).copy()
    arr[arr < cutoff] = 0.0
    return arr


def normalize_mode_shares(
    car_raw: np.ndarray,
    rail_raw: np.ndarray,
    air_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncate raw regressor predictions at zero and row-normalize across the three modes."""
    c = np.maximum(np.asarray(car_raw, dtype=float), 0.0)
    r = np.maximum(np.asarray(rail_raw, dtype=float), 0.0)
    a = np.maximum(np.asarray(air_raw, dtype=float), 0.0)
    denom = c + r + a
    valid = denom > 0.0
    out_c = np.divide(c, denom, out=np.zeros_like(c), where=valid)
    out_r = np.divide(r, denom, out=np.zeros_like(r), where=valid)
    out_a = np.divide(a, denom, out=np.zeros_like(a), where=valid)
    return out_c, out_r, out_a


def compute_route_date_emissions(
    distance_km: np.ndarray,
    flow_car: np.ndarray,
    flow_rail: np.ndarray,
    flow_air: np.ndarray,
    car_factor_g_pkm: np.ndarray,
    factor_set: str = "main",
) -> np.ndarray:
    """Compute route-date modeled emissions in tonnes CO2-equivalent.

    e_{rt} = 10^{-6} * d_r * (p_{rt,car} * f_{o,car} + p_{rt,rail} * f_{rail} + p_{rt,air} * f_{air}).
    """
    f = EMISSION_FACTOR_SETS[factor_set]
    d = np.asarray(distance_km, dtype=float)
    return 1e-6 * d * (
        np.asarray(flow_car, dtype=float) * np.asarray(car_factor_g_pkm, dtype=float)
        + np.asarray(flow_rail, dtype=float) * f["rail"]
        + np.asarray(flow_air, dtype=float) * f["air"]
    )


def compute_annual_route_accounting(
    route_panel: pd.DataFrame,
    route_col: str = "route_id",
    heat_col: str = "is_heatwave",
    routine_col: str = "is_routine",
    emission_col: str = "emissions_t",
) -> pd.DataFrame:
    """Aggregate route-date emissions into route-level G, D, and T totals.

    Conventions (Methods):
      - G_r = sum_{t in H_r} e_{rt}  (heatwave-coinciding emissions)
      - D_r = sum_{t in H_r} e_{rt} - n_{H,r} * mean_{t in R_r}(e_{rt})
        (heat-period excess emissions; negative route differences are retained)
      - T_r = sum_t e_{rt}  (all-day annual emissions across all available records)
    """
    hot = route_panel[route_panel[heat_col].astype(bool)]
    routine = route_panel[route_panel[routine_col].astype(bool)]

    g_hot = hot.groupby(route_col)[emission_col].agg(hot_sum="sum", n_hot="count")
    g_rout = routine.groupby(route_col)[emission_col].agg(routine_mean="mean", n_routine="count")
    g_all = route_panel.groupby(route_col)[emission_col].agg(all_sum="sum", n_all="count")

    out = g_all.join(g_hot, how="left").join(g_rout, how="left")
    out["hot_sum"] = out["hot_sum"].fillna(0.0)
    out["n_hot"] = out["n_hot"].fillna(0).astype(int)
    out["n_routine"] = out["n_routine"].fillna(0).astype(int)

    out["G_tonnes"] = out["hot_sum"]
    out["B_tonnes"] = out["n_hot"] * out["routine_mean"]
    # Routes with zero heatwave records contribute G = D = 0
    out.loc[out["n_hot"] == 0, "B_tonnes"] = 0.0
    out["D_tonnes"] = out["G_tonnes"] - out["B_tonnes"]
    out["T_tonnes"] = out["all_sum"]
    return out.reset_index()


def compute_city_conditional_indices(
    route_summary: pd.DataFrame,
    city_col: str,
    hot_mean_col: str = "e_bar_H",
    routine_mean_col: str = "e_bar_R",
) -> pd.DataFrame:
    """Compute city conditional-mean indices H, R and log-ratio emissions response 100 ln(H/R)."""
    agg = route_summary.groupby(city_col, as_index=False)[[hot_mean_col, routine_mean_col]].sum()
    agg = agg.rename(columns={hot_mean_col: "H", routine_mean_col: "R"})
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["emissions_response"] = np.where(
            (agg["H"] > 0.0) & (agg["R"] > 0.0),
            100.0 * np.log(agg["H"] / agg["R"]),
            np.nan,
        )
    return agg


def social_cost_usd(excess_tonnes: float, scc_usd_per_tonne: float = EPA_CENTRAL_SCC_USD_PER_TONNE) -> float:
    """Convert CO2-equivalent excess tonnes into social cost (USD) at the EPA central SCC."""
    return float(excess_tonnes) * float(scc_usd_per_tonne)
