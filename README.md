# Heatwaves and Intercity Human Mobility Carbon Inequality

Representative analysis code accompanying the manuscript:

> **Carbon inequality in heatwave-coinciding intercity mobility across China’s urban hierarchy**  
> Fengjue Huang, Junqing Tang, Pengjun Zhao\*, Wei Lyu, Jiaying Li, Jianlong Yang, Qiuchen Lu, Duo Li, Bao-jie He, and Naiyu Wang  
> \*Corresponding author: Pengjun Zhao (`pengjun.zhao@pku.edu.cn`)

---

## Overview

This repository provides clean, modular, self-contained Python implementations of the core analytical workflow used to quantify how extreme heatwaves reorganize intercity passenger-travel emissions across China’s urban hierarchy (366 prefecture-level cities, 2019–2024 historical observations and 2050 SSP5-8.5 projections).

Specifically, the representative codebase covers five stages corresponding directly to the **Methods** and **Supplementary Methods (S1–S6)**:

1. **Full-calendar heatwave reconstruction and event-timing diagnostics** (`src/01_heatwave_episodes.py`): identifies qualifying episodes of at least three consecutive calendar days strictly above 35 °C (plus 34 °C and 36 °C sensitivity thresholds) on uninterrupted annual city calendars, and constructs mutually exclusive lead (`-7` to `-4`, `-3` to `-2` days), lag (`+1` to `+7` days), and 14-day shifted diagnostic indicators.
2. **Harmonized well-to-wheel emissions accounting** (`src/02_emissions_accounting.py`): implements origin-specific fleet-weighted passenger-car factors, the harmonized `< 0.05` air-share cutoff, route-date emissions ($e_{rt}$), annual common-route totals ($G_y$, $D_y$, $T_y$), historical conditional city indices ($H$, $R$, $100\ln(H/R)$), US EPA (2023) social cost of carbon valuation (US$190 per tonne CO₂), and reference XGBoost mode-share/flow specifications.
3. **High-dimensional weighted fixed-effects route estimation** (`src/03_route_fixed_effects.py`): performs memory-bounded weighted alternating projections absorbing directed route, calendar date, and origin-by-ISO-week fixed effects (`tol = 1e-8`, verified orthogonality `< 5e-8`), stacked cross-year origin-clustered CR1 covariance ($V_0, V_1$), early-vs.-late period contrasts, and leave-one-year-out diagnostics.
4. **City income gradients and three-component route decomposition** (`src/04_city_gradients_decomposition.py`): estimates province-clustered WLS (routine-emissions weighted) and OLS (equal-city weighted) destination income gradients, applies Holm–Bonferroni family-wise multiplicity control, and executes the exact forward and reverse additive accounting decompositions (`travel intensity + route mix + within-route intensity`).
5. **Mid-century cooling and tiered electrification policy accounting** (`src/05_policy_scenarios_2050.py`): evaluates 0.5–2.0 °C ambient cooling across crossed flow-and-calendar matrices ($M_{00}, M_{c0}, M_{0c}, M_{cc}$), symmetric flow-vs.-calendar decomposition, city-tier ICE-to-BEV fleet transfers (80% and 100%), and joint scenario interactions.

---

## Repository Structure

```text
heatwaves-and-human-mobility/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── 01_heatwave_episodes.py
    ├── 02_emissions_accounting.py
    ├── 03_route_fixed_effects.py
    ├── 04_city_gradients_decomposition.py
    ├── 05_policy_scenarios_2050.py
    └── run_synthetic_verification.py
```

---

## Data Availability and Confidentiality Notice

**No raw, proprietary, or city-identifiable mobility datasets are hosted in this repository.**  
The original intercity mobility records (Baidu Migration daily origin–destination indices, 2019–2024; archived Tencent Location Services mode shares, 2015–2018) and municipal vehicle-fleet records are subject to third-party data-provider licensing and confidentiality restrictions and therefore cannot be publicly redistributed.

Publicly accessible upstream datasets used in the study can be obtained directly from their respective providers:

- **Historical station weather**: NOAA NCEI Global Summary of the Day (GSOD) archive (`https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/`)
- **Bias-corrected CMIP6 SSP5-8.5 climate projections**: National Tibetan Plateau Data Center (`https://data.tpdc.ac.cn/zh-hans/data/f3b17306-fd3e-423c-9084-83c41f9072c0`)
- **SSP gridded population projections**: Figshare (`https://doi.org/10.6084/m9.figshare.c.4605713`)
- **SSP gridded GDP projections**: Zenodo (`https://doi.org/10.5281/zenodo.5880037`)

Aggregated numerical values underlying the main figures, Extended Data figures, and Supplementary Tables are provided as **Source Data** accompanying the published article. Requests regarding the archived platform data schema and data dictionary may be directed to the corresponding author (`pengjun.zhao@pku.edu.cn`), subject to the providers’ access conditions.

---

## Installation and Verification

### 1. Environment setup

Tested with Python 3.10–3.12 on Windows and Linux. Install dependencies via `pip`:

```bash
git clone https://github.com/Fe6ruAr9-pku/heatwaves-and-human-mobility.git
cd heatwaves-and-human-mobility
pip install -r requirements.txt
```

### 2. Run the synthetic verification suite

Because confidential mobility records are not distributed here, we include a self-contained synthetic verification script (`src/run_synthetic_verification.py`) that generates purely synthetic toy city-network panels and verifies every core mathematical identity in under 5 seconds:

```bash
python src/run_synthetic_verification.py
```

Expected output:

```text
[PASS] 01_heatwave_episodes: run detection and zero lead-lag overlap verified.
[PASS] 02_emissions_accounting: fleet weighting, G/D/T, and EPA SCC verified.
[PASS] 03_route_fixed_effects: FE orthogonality < 5e-8, CR1 contrast = ...
[PASS] 04_city_gradients_decomposition: exact forward/reverse & WLS additivity verified.
[PASS] 05_policy_scenarios_2050: EV conservation, cooling split, and interaction verified.

ALL SYNTHETIC VERIFICATION CHECKS PASSED.
```

---

## Key Modeling Parameters

| Parameter | Primary value | Sensitivity / Notes |
| --- | --- | --- |
| Heatwave definition | $\ge 3$ consecutive days with $T_{\max} > 35\,^\circ\mathrm{C}$ | 34 °C and 36 °C on full annual calendars |
| Main emission factors (g CO₂-eq / pkm) | ICE 135, BEV 55, other NEV 77, rail 19, air 123 | High-labeled & low-labeled sets (Supplementary Table S2) |
| Harmonized air-share cutoff | $< 0.05 \rightarrow 0$ (no redistribution) | Applied uniformly across 2019–2024 |
| FE projection tolerance | `1e-8` (orthogonality check `< 5e-8`) | Directed route, date, origin-by-ISO-week FE |
| Multiplicity control | Holm–Bonferroni within predefined families | 216-test city registry; 84-test component registry |
| Social cost of carbon | US$190 / t CO₂ | US Environmental Protection Agency (2023) |

---

## License

This repository is released under the **MIT License** (see [`LICENSE`](LICENSE)).

## Citation

If you use or adapt this code, please cite the accompanying manuscript:

```bibtex
@article{huang2026heatwave_mobility_carbon,
  title   = {Carbon inequality in heatwave-coinciding intercity mobility across China's urban hierarchy},
  author  = {Huang, Fengjue and Tang, Junqing and Zhao, Pengjun and Lyu, Wei and Li, Jiaying and Yang, Jianlong and Lu, Qiuchen and Li, Duo and He, Bao-jie and Wang, Naiyu},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```
