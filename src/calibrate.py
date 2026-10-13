"""
Calibration module: scale appliance-level usage so that aggregate monthly
consumption matches ESCOM targets (rural ≈ 47 kWh, urban ≈ 160 kWh).

Method: iterative proportional fitting (one pass is usually enough).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from data_loader import generate_dataset


ESCOM_TARGETS = {"rural": 47.0, "urban": 160.0}


def monthly_avg_by_area(df: pd.DataFrame) -> dict[str, float]:
    """Return mean monthly kWh per household, by area."""
    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = (df.groupby(["household_id", "area", "month"])["consumption_kwh"]
                 .sum()
                 .reset_index())
    return monthly.groupby("area")["consumption_kwh"].mean().to_dict()


def find_scale_factors(
    n_households: int = 500,
    days: int = 180,          # longer window → captures seasons better
    seed: int = 1,
    max_iter: int = 3,
    tol: float = 0.02,        # stop when error < 2%
) -> dict[str, float]:
    """
    Iteratively find scale factors so aggregate monthly averages match
    ESCOM targets. Two-three passes usually converge.
    """
    scales = {area: 1.0 for area in ESCOM_TARGETS}

    for i in range(max_iter):
        df = generate_dataset(
            n_households=n_households, days=days, seed=seed,
            area_scales=scales,
        )
        current = monthly_avg_by_area(df)

        errors = {}
        new_scales = {}
        for area in ESCOM_TARGETS:
            target = ESCOM_TARGETS[area]
            got = current[area]
            errors[area] = (got - target) / target
            # multiplicative correction — stable in both directions
            new_scales[area] = scales[area] * (target / got)

        print(f"[iter {i+1}] current={ {k: round(v,1) for k,v in current.items()} } "
              f"errors={ {k: f'{v:+.1%}' for k,v in errors.items()} }")

        if all(abs(e) < tol for e in errors.values()):
            print(f"[iter {i+1}] converged ✓")
            scales = new_scales
            break

        scales = new_scales

    return scales


if __name__ == "__main__":
    scales = find_scale_factors()
    print("Scale factors to apply:")
    for area, s in scales.items():
        print(f"  {area:6s} → x{s:.3f}")