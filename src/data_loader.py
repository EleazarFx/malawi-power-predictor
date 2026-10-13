"""
Synthetic Malawi household electricity consumption generator.

Grounded in the ESCOM Load Forecast Report 2024-2034 (July 2025).
Key assumptions taken directly from the report:
  - Rural household: ~47 kWh/month  (§4.3, Table 6)
  - Urban household: ~160 kWh/month (§4.3)
  - Household size: 4.3 (2023) declining 0.02/yr (Table 17)
  - Rainy season: Nov-Apr
  - Load shedding: drought-driven, more frequent in dry season
  - Appliance mix: Table 6 (rural) + Table 14 (general)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# 1. Appliance library  (watts, typical daily hours, penetration by area)
# ---------------------------------------------------------------------------
# Sources: ESCOM Table 6 (rural appliances) and Table 14 (general).
# Penetration values are our estimates for realism — refine later with survey data.

@dataclass
class Appliance:
    name: str
    watts: float
    hours_per_day: float
    penetration_urban: float   # 0-1
    penetration_rural: float
    peak_weights: dict[int, float]   # {hour: relative strength 0.1–1.0}
    seasonal_bias: str = "none"  # 'rainy' boosts, 'dry' boosts, or 'none'


APPLIANCES = [
    Appliance("led_bulb",        9,    5.0,  0.95, 0.90,
              {18: 1.0, 19: 1.0, 20: 1.0, 21: 0.9, 22: 0.7,
               23: 0.4, 0: 0.2, 5: 0.6, 6: 0.9}),
    Appliance("fluorescent_tube",15,   4.0,  0.40, 0.60,
              {18: 0.9, 19: 1.0, 20: 1.0, 21: 0.9, 22: 0.6}),
    Appliance("tv_25in",         150,  4.0,  0.55, 0.45,
              {18: 0.8, 19: 1.0, 20: 1.0, 21: 0.9, 22: 0.6}),
    Appliance("tv_32in_led",     41,   4.0,  0.40, 0.05,
              {18: 0.8, 19: 1.0, 20: 1.0, 21: 0.9, 22: 0.6}),
    Appliance("radio",           4,    5.0,  0.60, 0.80,
              {6: 1.0, 7: 1.0, 8: 0.7, 12: 0.4, 13: 0.4,
               18: 0.8, 19: 0.8, 20: 0.7}),
    Appliance("laptop",          35,   2.0,  0.35, 0.05,
              {8: 0.4, 9: 0.5, 10: 0.3, 19: 0.8, 20: 1.0, 21: 0.9, 22: 0.6}),
    Appliance("phone_charging",  5,    3.0,  0.90, 0.85,
              {7: 0.8, 8: 0.9, 20: 1.0, 21: 1.0, 22: 0.9, 23: 0.6}),
    Appliance("kettle",          2200, 0.2,  0.50, 0.30,
              {6: 1.0, 7: 0.7, 12: 0.3, 13: 0.3, 18: 0.8, 19: 0.9}),  # <-- lunch reduced
    Appliance("iron",            1000, 0.3,  0.70, 0.50,
              {6: 0.6, 7: 0.9, 8: 0.7, 17: 0.8, 18: 0.9, 19: 0.7}),
    Appliance("hotplate",        1000, 1.0,  0.15, 0.00,
              {6: 0.6, 7: 0.9, 12: 0.3, 13: 0.3, 18: 1.0, 19: 1.0, 20: 0.6}),
    Appliance("fridge_small",    100,  6.0,  0.60, 0.35,
              {h: 0.25 for h in range(24)}),   # ~flat, always on
    Appliance("fridge_large",    200,  8.0,  0.25, 0.00,
              {h: 0.33 for h in range(24)}),
    Appliance("fan",             60,   4.0,  0.35, 0.10,
              {12: 0.4, 13: 0.6, 14: 0.7, 15: 0.6, 20: 1.0, 21: 0.9, 22: 0.5},
              seasonal_bias="dry"),
    Appliance("ac_unit",         1200, 3.0,  0.05, 0.00,
              {13: 0.4, 14: 0.6, 15: 0.6, 20: 1.0, 21: 0.9, 22: 0.5},
              seasonal_bias="dry"),
    Appliance("water_heater",    2000, 1.0,  0.08, 0.00,
              {6: 0.9, 7: 1.0, 19: 1.0, 20: 0.9}),
]

# ---------------------------------------------------------------------------
# 2. Utility: build per-hour usage probability for an appliance
# ---------------------------------------------------------------------------

def _hour_usage_probability(appliance: Appliance, hour: int,
                            rng: np.random.Generator) -> float:
    base_prob = appliance.hours_per_day / 24.0
    weight = appliance.peak_weights.get(hour)

    if weight is not None:
        prob = min(0.95, base_prob * weight * 24 / max(len(appliance.peak_weights), 1) * rng.uniform(0.85, 1.15))
    else:
        prob = base_prob * rng.uniform(0.05, 0.2)

    return float(np.clip(prob, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Household class — one simulated home
# ---------------------------------------------------------------------------

@dataclass
class Household:
    household_id: int
    area: str
    region: str
    household_size: int
    appliances: List[Appliance] = field(default_factory=list)
    area_scale: float = 1.0   

    @classmethod
    def sample(cls, household_id, area, region, rng,
               area_scale: float = 1.0):    
        size = int(rng.choice([2, 3, 4, 5, 6], p=[0.10, 0.20, 0.30, 0.25, 0.15]))
        owned = []
        for app in APPLIANCES:
            p = app.penetration_urban if area == "urban" else app.penetration_rural
            if rng.random() < p:
                owned.append(app)
        if not owned:
            owned = [APPLIANCES[0]]
        return cls(household_id, area, region, size, owned, area_scale)  

    def simulate_hour(self, hour: int, is_weekend: bool, is_rainy: bool,
                      rng: np.random.Generator) -> float:
        """Return kWh consumed in this hour by this household."""
        kwh = 0.0
        # Larger households use more of everything (diminishing returns)
        size_factor = 1.0 + 0.15 * (self.household_size - 3)

        for app in self.appliances:
            p_on = _hour_usage_probability(app, hour, rng)

            # Weekend: more daytime TV / cooking / ironing
            if is_weekend and hour in range(9, 18):
                p_on *= 1.3

            # Seasonal: fans/AC run more in hot dry season
            if app.seasonal_bias == "dry" and not is_rainy:
                p_on *= 1.4
            elif app.seasonal_bias == "rainy" and is_rainy:
                p_on *= 1.3

            is_on = rng.random() < p_on
            if is_on:
                # small continuous variation of actual draw
                kwh += app.watts * rng.uniform(0.85, 1.15) / 1000.0

        # NEW: scale up so the household hits its ESCOM-implied average
        return kwh * size_factor * self.area_scale



# ---------------------------------------------------------------------------
# 4. Main generation function
# ---------------------------------------------------------------------------
"""
Build a synthetic hourly Malawi household consumption dataset.
Returns a long-format DataFrame:
household_id, timestamp, region, area, household_size,
consumption_kwh, is_weekend, is_rainy_season, load_shed,
hour, day_of_week, month
"""
def generate_dataset(
    n_households: int = 200,
    start: str = "2024-01-01",
    days: int = 365,
    seed: int = 42,
    area_scales: dict | None = None,    
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if area_scales is None:
        area_scales = {"urban": 1.0, "rural": 1.0}   

    areas = rng.choice(["urban", "rural"], size=n_households, p=[0.30, 0.70])
    regions = rng.choice(["North", "Central", "South"],
        size=n_households, p=[0.10, 0.36, 0.54])
    households = [
        Household.sample(i, areas[i], regions[i], rng,
            area_scale=area_scales[areas[i]])   
        for i in range(n_households)
    ]
   

    # --- Time index ---
    timestamps = pd.date_range(start, periods=days * 24, freq="h")

    # --- Load shedding schedule (report says dry season is worse) ---
    # ~6% of hours in dry season, ~2% in rainy season.
    load_shed_mask = np.zeros(len(timestamps), dtype=int)
    for i, ts in enumerate(timestamps):
        is_rainy = ts.month in (11, 12, 1, 2, 3, 4)
        p_shed = 0.02 if is_rainy else 0.06
        if rng.random() < p_shed:
            load_shed_mask[i] = 1

    # --- Simulate each household ---
    records = []
    for hh in households:
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            is_weekend = ts.dayofweek >= 5
            is_rainy = ts.month in (11, 12, 1, 2, 3, 4)

            kwh = hh.simulate_hour(hour, is_weekend, is_rainy, rng)

            # Apply load shedding: consumption drops to ~0
            if load_shed_mask[i]:
                kwh = 0.0

            records.append((
                hh.household_id, ts, hh.region, hh.area, hh.household_size,
                round(kwh, 4), int(is_weekend), int(is_rainy),
                int(load_shed_mask[i]), hour, ts.dayofweek, ts.month,
            ))

    df = pd.DataFrame(records, columns=[
        "household_id", "timestamp", "region", "area", "household_size",
        "consumption_kwh", "is_weekend", "is_rainy_season", "load_shed",
        "hour", "day_of_week", "month",
    ])
    return df


# ---------------------------------------------------------------------------
# 5. Quality check helper
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare simulated averages against ESCOM report targets."""
    df = df.copy()
    df["day"] = df["timestamp"].dt.date

    daily = (df.groupby(["household_id", "area", "day"])["consumption_kwh"]
               .sum()
               .reset_index())
    monthly = daily.copy()
    monthly["month"] = pd.to_datetime(monthly["day"]).dt.month
    monthly = monthly.groupby(["household_id", "area", "month"])["consumption_kwh"].sum()
    monthly = monthly.reset_index()

    summary = (monthly.groupby("area")["consumption_kwh"]
                      .mean()
                      .reset_index()
                      .rename(columns={"consumption_kwh": "avg_monthly_kWh"}))
    summary["target_from_ESCOM"] = summary["area"].map({
        "rural": 47.0,   # Table 6 / §4.3
        "urban": 160.0,  # §4.3
    })
    summary["error_%"] = ((summary["avg_monthly_kWh"] - summary["target_from_ESCOM"])
                          / summary["target_from_ESCOM"] * 100).round(1)
    return summary


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)

    # 1. Find scale factors
    from calibrate import find_scale_factors
    scales = find_scale_factors()
    print("Calibration scales:", {k: round(v, 3) for k, v in scales.items()})

    # 2. Generate the full dataset with those scales
    df = generate_dataset(n_households=200, days=365, seed=999,
                          area_scales=scales)
    out = "data/raw/malawi_households_hourly.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows -> {out}")

    print("\n--- Validation vs ESCOM targets ---")
    print(dataset_summary(df).to_string(index=False))