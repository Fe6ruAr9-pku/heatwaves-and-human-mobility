"""Full-calendar heatwave reconstruction and mutually exclusive event-timing indicators.

Implements the heatwave definition and lead/lag/shifted diagnostic variables described in
Methods ("Heatwave exposure and income classification") and Supplementary Methods S1 & S5:
  - A heatwave is defined as at least `min_consecutive_days` (default: 3) consecutive
    calendar days with daily maximum temperature strictly above `threshold_c` (default:
    35.0 deg C; sensitivities: 34.0 and 36.0 deg C).
  - Episodes are reconstructed on full annual city calendars before subsetting to warm-
    season mobility records so that seasonal boundaries do not truncate ongoing runs.
  - Gaps in calendar dates break consecutive runs.
  - Lead (-7 to -4, -3 to -2 days) and lag (+1 to +7 days) indicators are mutually
    exclusive with heatwave days, with nearer upcoming onsets taking priority over
    earlier leads and prior-episode lags; day -1 remains in the reference category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_heatwave_runs(
    city_weather: pd.DataFrame,
    threshold_c: float = 35.0,
    min_consecutive_days: int = 3,
    city_col: str = "city",
    date_col: str = "date",
    tmax_col: str = "tmax_c",
) -> pd.Series:
    """Flag every calendar day belonging to a qualifying heatwave run.

    Parameters
    ----------
    city_weather : pd.DataFrame
        Long table with city identifier, calendar date, and daily maximum temperature
        in degrees Celsius.
    threshold_c : float, default 35.0
        Strict temperature threshold (daily max strictly greater than `threshold_c`).
    min_consecutive_days : int, default 3
        Minimum consecutive calendar days required to qualify as a heatwave episode.

    Returns
    -------
    pd.Series
        Integer series (0/1) aligned to `city_weather.index`.
    """
    df = city_weather[[city_col, date_col, tmax_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    flags = pd.Series(0, index=df.index, dtype=np.int8)

    for _, group in df.groupby(city_col, sort=False):
        g = group.sort_values(date_col)
        dates = g[date_col].to_numpy()
        hot = (g[tmax_col].to_numpy(dtype=float) > threshold_c) & np.isfinite(
            g[tmax_col].to_numpy(dtype=float)
        )
        n = len(g)
        if n == 0:
            continue

        # Identify contiguous calendar runs (1-day step) where hot is True
        run_flag = np.zeros(n, dtype=np.int8)
        start = 0
        while start < n:
            if not hot[start]:
                start += 1
                continue
            end = start + 1
            while end < n and hot[end] and (dates[end] - dates[end - 1]) == np.timedelta64(1, "D"):
                end += 1
            if (end - start) >= min_consecutive_days:
                run_flag[start:end] = 1
            start = end

        flags.loc[g.index] = run_flag

    return flags


def build_event_timing_indicators(
    city_calendar: pd.DataFrame,
    hw_col: str = "hw35",
    city_col: str = "city",
    date_col: str = "date",
) -> pd.DataFrame:
    """Construct calendar-aware, mutually exclusive lead, lag, and 14-day shifted indicators.

    Priority rules (Supplementary Methods S5):
      1. Active heatwave days (`hw_col == 1`) take priority over all lead/lag indicators.
      2. If windows overlap across recurrent episodes, the nearer upcoming onset
         (`lead3_2`) takes priority over `lead7_4`, and any upcoming onset takes priority
         over a previous episode's lag (`lag1_7`).
      3. Day -1 prior to onset remains in the reference category.
      4. `shift14` evaluates the heatwave status 14 calendar days ahead as a diagnostic
         timing check.
    """
    out = city_calendar.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    for name in ("lead7_4", "lead3_2", "lag1_7", "shift14"):
        out[name] = np.nan

    for _, g in out.groupby(city_col, sort=False):
        g_sorted = g.sort_values(date_col)
        s = pd.Series(g_sorted[hw_col].to_numpy(dtype=float), index=g_sorted[date_col])
        prev_day = s.reindex(s.index - pd.Timedelta(days=1)).to_numpy()
        next_day = s.reindex(s.index + pd.Timedelta(days=1)).to_numpy()

        starts = s.index[(s.to_numpy() == 1.0) & (prev_day == 0.0)]
        ends = s.index[(s.to_numpy() == 1.0) & (next_day == 0.0)]

        specs = [
            ("lead7_4", range(-7, -3), starts, range(0, 8)),
            ("lead3_2", range(-3, -1), starts, range(0, 4)),
            ("lag1_7", range(1, 8), ends, range(-7, 1)),
        ]
        for name, offsets, events, required_offsets in specs:
            target_dates = {d + pd.Timedelta(days=k) for d in events for k in offsets}
            vals = np.array([float(d in target_dates) for d in s.index], dtype=float)
            known = np.ones(len(s), dtype=bool)
            for k in required_offsets:
                known &= np.isfinite(s.reindex(s.index + pd.Timedelta(days=k)).to_numpy())
            vals[~known] = np.nan
            vals[s.to_numpy() == 1.0] = 0.0
            out.loc[g_sorted.index, name] = vals

        out.loc[g_sorted.index, "shift14"] = s.reindex(
            s.index + pd.Timedelta(days=14)
        ).to_numpy()

    # Enforce strict mutual exclusivity across overlapping recurrent episodes
    out.loc[out["lead3_2"].eq(1.0), "lead7_4"] = 0.0
    out.loc[out["lead7_4"].eq(1.0) | out["lead3_2"].eq(1.0), "lag1_7"] = 0.0
    return out
