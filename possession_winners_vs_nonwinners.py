import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Load the original CSV files
# -------------------------------------------------
team_stats = pd.read_csv("match_team_stats.csv")
matches = pd.read_csv("matches.csv")

print("match_team_stats shape :", team_stats.shape)
print("matches shape          :", matches.shape)
print()

# -------------------------------------------------
# 2. Data wrangling – create Winner / Non-Winner label
# -------------------------------------------------
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["away_score"] > row["home_score"]:
        return "away_win"
    else:
        return "draw"

matches["result"] = matches.apply(get_result, axis=1)

df = team_stats.merge(
    matches[["match_id", "home_team_id", "away_team_id", "result"]],
    on="match_id",
    how="left"
)

def label_team(row):
    if row["result"] == "draw":
        return "Non-Winner"
    if row["result"] == "home_win" and row["team_id"] == row["home_team_id"]:
        return "Winner"
    if row["result"] == "away_win" and row["team_id"] == row["away_team_id"]:
        return "Winner"
    return "Non-Winner"

df["outcome"] = df.apply(label_team, axis=1)

print("Outcome counts:")
print(df["outcome"].value_counts())
print()

# -------------------------------------------------
# 3. Data preparation
# -------------------------------------------------
winner_poss    = df.loc[df["outcome"] == "Winner", "possession_pct"].dropna()
nonwinner_poss = df.loc[df["outcome"] == "Non-Winner", "possession_pct"].dropna()

print(f"Winner observations     : {len(winner_poss)}")
print(f"Non-Winner observations : {len(nonwinner_poss)}")
print()

# -------------------------------------------------
# 4. Descriptive statistics
# -------------------------------------------------
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print("\nWinner possession %:")
print(winner_poss.describe())
print("\nNon-Winner possession %:")
print(nonwinner_poss.describe())

# -------------------------------------------------
# 5. 95% Confidence Intervals
# -------------------------------------------------
def mean_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

w_mean, w_lo, w_hi = mean_ci(winner_poss)
nw_mean, nw_lo, nw_hi = mean_ci(nonwinner_poss)

print("\n" + "=" * 60)
print("95% CONFIDENCE INTERVALS")
print("=" * 60)
print(f"Winner     mean = {w_mean:.2f}%   95% CI [{w_lo:.2f}, {w_hi:.2f}]")
print(f"Non-Winner mean = {nw_mean:.2f}%   95% CI [{nw_lo:.2f}, {nw_hi:.2f}]")

# -------------------------------------------------
# 6. Two-sample t-test
# -------------------------------------------------
levene_stat, levene_p = stats.levene(winner_poss, nonwinner_poss)
equal_var = levene_p > 0.05

t_stat, p_value = stats.ttest_ind(winner_poss, nonwinner_poss, equal_var=equal_var)

print("\n" + "=" * 60)
print("TWO-SAMPLE t-TEST")
print("=" * 60)
print(f"Levene's test p-value = {levene_p:.4f} → equal_var = {equal_var}")
print(f"t-statistic = {t_stat:.3f}")
print(f"p-value     = {p_value:.6f}")

if p_value < 0.05:
    print("\nConclusion: There is a statistically significant difference")
    print("in average possession between winners and non-winners (α = 0.05).")
else:
    print("\nConclusion: There is NO statistically significant difference")
    print("in average possession between winners and non-winners (α = 0.05).")

# -------------------------------------------------
# 7. Visualisation
# -------------------------------------------------
plot_df = pd.DataFrame({
    "Possession %": pd.concat([winner_poss, nonwinner_poss]),
    "Group": ["Winners"] * len(winner_poss) + ["Non-Winners\n(Draw or Loss)"] * len(nonwinner_poss)
})

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=plot_df,
    x="Group",
    y="Possession %",
    palette=["#2ecc71", "#e74c3c"],
    width=0.5
)

plt.title("Possession % – Winning Teams vs Non-Winning Teams\nFIFA World Cup 2026", fontsize=14)
plt.ylabel("Possession %", fontsize=12)
plt.xlabel("")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("possession_winners_vs_nonwinners.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFigure saved as: possession_winners_vs_nonwinners.png")
print("Analysis complete.")