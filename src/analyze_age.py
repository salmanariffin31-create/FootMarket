import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load training dataset only
df = pd.read_csv("data/processed/splits/train.csv")

# Group market values by age
age_analysis = df.groupby("age")["market_value"].agg(
    ["mean", "median", "count"]
).reset_index()

print(age_analysis.to_string(index=False))

# Plot market value by age
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=age_analysis,
    x="age",
    y="mean",
    label="Mean market value",
    marker="o"
)

sns.lineplot(
    data=age_analysis,
    x="age",
    y="median",
    label="Median market value",
    marker="o"
)

plt.title("Football Player Market Value by Age")
plt.xlabel("Player Age")
plt.ylabel("Market Value (EUR)")
plt.ticklabel_format(style="plain", axis="y")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()