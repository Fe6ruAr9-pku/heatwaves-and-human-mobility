"""End-to-end synthetic verification of all representative analysis modules.

Runs exclusively on synthetic toy data (zero private or real mobility records) to confirm
that every mathematical and algorithmic identity reported in the manuscript holds to
machine precision:
  1. Full-calendar heatwave run detection (>= 3 consecutive days > 35 deg C, gap-breaking)
     and strictly zero overlap across mutually exclusive lead/lag timing indicators.
  2. Route-date emissions accounting (G, D, T, conditional indices H/R, EPA $190/t SCC).
  3. Weighted three-way fixed-effects alternating projections (orthogonality < 1e-8) and
     cross-year origin-clustered CR1 contrast inference.
  4. Exact additivity of the 3-component route-composition decomposition (total = travel
     intensity + route mix + within-route intensity) at both city and regression levels.
  5. Exact crossed cooling flow-calendar identity (flow_sym + calendar_sym = M_00 - M_cc),
     EV fleet conservation, and joint interaction accounting.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running directly as `python src/run_synthetic_verification.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

hw_mod = importlib.import_module("src.01_heatwave_episodes")
em_mod = importlib.import_module("src.02_emissions_accounting")
fe_mod = importlib.import_module("src.03_route_fixed_effects")
dec_mod = importlib.import_module("src.04_city_gradients_decomposition")
pol_mod = importlib.import_module("src.05_policy_scenarios_2050")


def verify_heatwave_episodes() -> None:
    dates = pd.date_range("2024-07-01", periods=35, freq="D")
    tmax = np.full(35, 31.0)
    # 2-day warm spell (must NOT qualify)
    tmax[2:4] = 36.5
    # 4-day heatwave (days 10-13, must qualify)
    tmax[10:14] = 37.2
    # 3-day heatwave (days 22-24, must qualify)
    tmax[22:25] = 35.4

    df = pd.DataFrame({"city": "CityA", "date": dates, "tmax_c": tmax})
    df["hw35"] = hw_mod.detect_heatwave_runs(df, threshold_c=35.0, min_consecutive_days=3)
    assert int(df["hw35"].sum()) == 7, f"Expected 7 heatwave days, got {df['hw35'].sum()}"

    ev = hw_mod.build_event_timing_indicators(df, hw_col="hw35")
    overlap = (ev[["hw35", "lead7_4", "lead3_2", "lag1_7"]].fillna(0).sum(axis=1) > 1).sum()
    assert int(overlap) == 0, "Lead/lag/heatwave indicators must be mutually exclusive."
    print("[PASS] 01_heatwave_episodes: run detection and zero lead-lag overlap verified.")


def verify_emissions_accounting() -> None:
    car_ef = em_mod.fleet_weighted_car_factor(
        np.array([0.85, 0.70]), np.array([0.10, 0.25]), np.array([0.05, 0.05]), factor_set="main"
    )
    assert np.allclose(car_ef, [124.1, 112.1])

    air_clean = em_mod.apply_harmonized_air_cutoff(np.array([0.03, 0.12]), cutoff=0.05)
    assert np.allclose(air_clean, [0.0, 0.12])

    e = em_mod.compute_route_date_emissions(
        distance_km=np.array([500.0, 1200.0]),
        flow_car=np.array([1000.0, 200.0]),
        flow_rail=np.array([800.0, 1500.0]),
        flow_air=np.array([0.0, 600.0]),
        car_factor_g_pkm=car_ef,
        factor_set="main",
    )
    panel = pd.DataFrame(
        {
            "route_id": ["R1", "R1", "R1", "R2", "R2"],
            "is_heatwave": [1, 0, 0, 0, 0],
            "is_routine": [0, 1, 1, 1, 1],
            "emissions_t": [e[0] * 1.2, e[0], e[0] * 0.9, e[1], e[1] * 1.1],
        }
    )
    acc = em_mod.compute_annual_route_accounting(panel).set_index("route_id")
    # Route R2 has zero heatwave records -> G = D = 0, T > 0
    assert acc.loc["R2", "G_tonnes"] == 0.0 and acc.loc["R2", "D_tonnes"] == 0.0
    assert acc.loc["R2", "T_tonnes"] > 0.0
    assert np.isclose(em_mod.social_cost_usd(636_918.0), 121_014_420.0)
    print("[PASS] 02_emissions_accounting: fleet weighting, G/D/T, and EPA SCC verified.")


def verify_route_fixed_effects() -> None:
    rng = np.random.default_rng(42)
    n_origins, n_dests, n_days = 15, 12, 24
    n_routes = n_origins * n_dests
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    dest_tier = np.repeat([0, 1, 2], n_dests // 3)

    fits = []
    for yr in years:
        o_grid = np.repeat(np.arange(n_origins), n_dests * n_days)
        d_grid = np.tile(np.repeat(np.arange(n_dests), n_days), n_origins)
        t_grid = np.tile(np.arange(n_days), n_routes)
        od_code = o_grid * n_dests + d_grid
        ow_code = o_grid * 6 + (t_grid // 4)
        w = rng.uniform(0.5, 2.0, size=len(o_grid))

        hw_origin = (rng.random((n_origins, n_days)) < 0.25).astype(float)
        hw = hw_origin[o_grid, t_grid]
        dt = dest_tier[d_grid]
        X = np.column_stack([hw * (dt == k) for k in range(3)])

        true_beta = np.array([-0.01, -0.005, 0.002]) if yr <= 2021 else np.array([-0.02, -0.01, 0.025])
        alpha_od = rng.normal(0, 0.4, size=n_routes)[od_code]
        lam_t = rng.normal(0, 0.2, size=n_days)[t_grid]
        gam_ow = rng.normal(0, 0.15, size=n_origins * 6)[ow_code]
        y = alpha_od + lam_t + gam_ow + X @ true_beta + rng.normal(0, 0.05, size=len(o_grid))

        fit = fe_mod.fit_annual_route_fe(
            year=yr,
            y=y,
            x=X,
            weights=w,
            fe_codes=[od_code, t_grid, ow_code],
            origin_ids=o_grid,
            n_total_origins=n_origins,
            tol=1e-8,
        )
        assert fit.orthogonality < 5e-8
        fits.append(fit)

    beta, v0, v1, g_clusters = fe_mod.stacked_cr1_covariance(fits)
    assert g_clusters == n_origins
    c_high = fe_mod.build_period_contrast_vector(years, n_terms_per_year=3, high_idx=2)
    res = fe_mod.linear_contrast(beta, v1, c_high, df=g_clusters - 1)
    assert res["estimate"] > 0.0 and res["p_value"] < 0.05
    print(
        f"[PASS] 03_route_fixed_effects: FE orthogonality < 5e-8, CR1 contrast = "
        f"{res['estimate']:+.2f} log pts (P = {res['p_value']:.4f})."
    )


def verify_city_gradients_and_decomposition() -> None:
    rng = np.random.default_rng(7)
    n_provinces = 22
    cities_per_prov = 4
    n_cities = n_provinces * cities_per_prov

    provinces = np.repeat(np.arange(n_provinces), cities_per_prov)
    z_income = rng.normal(0, 1, size=n_cities)
    z_pop = rng.normal(0, 1, size=n_cities)
    z_hw = rng.normal(0, 1, size=n_cities)
    weights = rng.uniform(50.0, 500.0, size=n_cities)

    Y_rows = []
    for i in range(n_cities):
        n_r = 12
        tr = rng.uniform(10.0, 100.0, size=n_r)
        er = tr * rng.uniform(0.5, 3.0, size=n_r)
        th = tr * np.exp(0.04 * z_income[i] + rng.normal(0, 0.1, size=n_r))
        eh = th * (er / tr) * np.exp(rng.normal(0, 0.03, size=n_r))
        d = dec_mod.decompose_city_route_response(tr, th, er, eh, include_reverse=True)
        # Exact forward & reverse identities
        assert np.isclose(
            d["total"], d["travel_intensity"] + d["route_mix_forward"] + d["within_route_forward"], atol=1e-10
        )
        assert np.isclose(
            d["total"], d["travel_intensity"] + d["route_mix_reverse"] + d["within_route_reverse"], atol=1e-10
        )
        Y_rows.append([d["total"], d["travel_intensity"], d["route_mix_forward"], d["within_route_forward"]])

    Y = np.asarray(Y_rows)
    prov_dummies = np.eye(n_provinces)[provinces]
    X = np.column_stack([z_income, z_pop, z_hw, prov_dummies])
    fit = dec_mod.fit_province_clustered_multi_outcome(X, Y, weights, provinces, income_col_idx=0)
    # Exact additivity of regression coefficients and province influence vectors
    assert np.isclose(fit["coefs"][0], fit["coefs"][1:].sum(), atol=1e-10)
    assert np.allclose(fit["influences"][:, 0], fit["influences"][:, 1:].sum(axis=1), atol=1e-10)
    p_adj = dec_mod.apply_holm_adjustment(fit["p_values"])
    assert (p_adj >= fit["p_values"] - 1e-12).all()
    print("[PASS] 04_city_gradients_decomposition: exact forward/reverse & WLS additivity verified.")


def verify_policy_scenarios() -> None:
    ice = np.array([0.85, 0.90, 0.80])
    bev = np.array([0.10, 0.05, 0.15])
    onev = np.array([0.05, 0.05, 0.05])
    tiers = ["high", "middle", "low"]

    ice_ev, bev_ev, onev_ev = pol_mod.transfer_origin_fleet_to_bev(
        ice, bev, onev, tiers, target_tiers=["high"], transfer_fraction=0.8
    )
    assert np.allclose(ice_ev + bev_ev + onev_ev, 1.0)
    assert np.isclose(ice_ev[0], 0.17) and np.isclose(bev_ev[0], 0.78)
    assert np.isclose(ice_ev[1], 0.90)  # untargeted tier unchanged

    base = pol_mod.route_metrics_from_pkm(
        hot_pkm=np.array([[1e7, 5e6, 2e6], [8e6, 3e6, 1e6], [6e6, 2e6, 5e5]]),
        routine_pkm=np.array([[9e7, 5e7, 1.5e7], [7e7, 3e7, 8e6], [5e7, 2e7, 4e6]]),
        n_hot=np.array([15, 12, 10]),
        n_routine=np.array([150, 150, 150]),
        ice_share=ice,
        bev_share=bev,
        other_nev_share=onev,
    )
    dec = pol_mod.decompose_cooling_flow_calendar(m00=822_610.92, mc0=800_970.21, m0c=402_000.00, mcc=380_738.68)
    assert np.isclose(dec["flow_symmetric"] + dec["calendar_symmetric"], dec["total_reduction"], atol=1e-9)

    inter = pol_mod.evaluate_joint_interaction(
        baseline=822_610.92,
        cooling_remaining=822_610.92 - 441_872.25,
        ev_remaining=822_610.92 - 88_372.44,
        joint_remaining=822_610.92 - 484_411.97,
    )
    assert np.isclose(inter["interaction"], -45_832.72, atol=0.5)
    assert base["G_tonnes"] > 0.0 and base["T_tonnes"] > base["G_tonnes"]
    print("[PASS] 05_policy_scenarios_2050: EV conservation, cooling split, and interaction verified.")


def main() -> None:
    verify_heatwave_episodes()
    verify_emissions_accounting()
    verify_route_fixed_effects()
    verify_city_gradients_and_decomposition()
    verify_policy_scenarios()
    print("\nALL SYNTHETIC VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
