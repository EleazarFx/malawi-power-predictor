import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/malawi_households_hourly.csv", parse_dates=["timestamp"])

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Average hourly profile, urban vs rural
for area in ["urban", "rural"]:
    sub = df[df.area == area]
    sub.groupby("hour")["consumption_kwh"].mean().plot(ax=axes[0], label=area)
axes[0].set_title("Average daily profile")
axes[0].set_xlabel("Hour of day")
axes[0].set_ylabel("Mean kWh")
axes[0].legend()

# 2. Weekday vs weekend (urban only)
urban = df[df.area == "urban"]
for wd, lbl in [(0, "weekday"), (1, "weekend")]:
    urban[urban.is_weekend == wd].groupby("hour")["consumption_kwh"].mean().plot(
        ax=axes[1], label=lbl)
axes[1].set_title("Urban: weekday vs weekend")
axes[1].legend()

# 3. Rainy vs dry season
for rainy, lbl in [(1, "rainy (Nov-Apr)"), (0, "dry (May-Oct)")]:
    df[df.is_rainy_season == rainy].groupby("hour")["consumption_kwh"].mean().plot(
        ax=axes[2], label=lbl)
axes[2].set_title("Rainy vs dry season")
axes[2].legend()

plt.tight_layout()
plt.savefig("data/processed/sanity_check_v2.png", dpi=100, bbox_inches="tight")
plt.show()