# HIT140 - Foundation of Data Science
# Objective 1: Analytic Question
#
# Question:
# Is there a statistically significant difference in the average
# number of goals scored per match between winning teams and
# losing teams in the FIFA World Cup 2026?

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "FIFA_World_Cup_2026_Objective1_Dataset.xlsx"

data = pd.read_excel(
    file_path,
    sheet_name="Objective_1_Data"
)

print("Dataset loaded successfully.")
print("Number of rows:", len(data))


# ============================================================
# 2. PREPARE DATA
# ============================================================

analysis_data = data[
    ["Winner_Goals", "Loser_Goals"]
].dropna()

winner_goals = analysis_data["Winner_Goals"]
loser_goals = analysis_data["Loser_Goals"]

print("\nNumber of matches analysed:", len(analysis_data))


# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

winner_mean = winner_goals.mean()
winner_sd = winner_goals.std(ddof=1)

loser_mean = loser_goals.mean()
loser_sd = loser_goals.std(ddof=1)

print("\n--- DESCRIPTIVE STATISTICS ---")

print("\nWinning Teams")
print("Mean:", round(winner_mean, 2))
print("Standard deviation:", round(winner_sd, 2))

print("\nLosing Teams")
print("Mean:", round(loser_mean, 2))
print("Standard deviation:", round(loser_sd, 2))


# ============================================================
# 4. CALCULATE MEAN DIFFERENCE
# ============================================================

difference = winner_goals - loser_goals

mean_difference = difference.mean()

print("\nMean difference (Winner - Loser):",
      round(mean_difference, 2),
      "goals per match")


# ============================================================
# 5. HYPOTHESES
# ============================================================

print("\n--- HYPOTHESES ---")

print("""
H0: There is no statistically significant difference
    in average goals scored by winning and losing teams.

H1: There is a statistically significant difference
    in average goals scored by winning and losing teams.
""")


# ============================================================
# 6. PAIRED-SAMPLES T-TEST
# ============================================================

# The two scores come from the same match.
# Therefore, the observations are paired.

t_statistic, p_value = stats.ttest_rel(
    winner_goals,
    loser_goals
)

degrees_of_freedom = len(difference) - 1

print("--- PAIRED-SAMPLES T-TEST ---")
print("t-statistic:", round(t_statistic, 2))
print("Degrees of freedom:", degrees_of_freedom)
print("p-value:", p_value)


# ============================================================
# 7. 95% CONFIDENCE INTERVAL
# ============================================================

confidence_level = 0.95
alpha = 1 - confidence_level

standard_error = (
    difference.std(ddof=1) /
    np.sqrt(len(difference))
)

critical_value = stats.t.ppf(
    1 - alpha / 2,
    degrees_of_freedom
)

margin_of_error = critical_value * standard_error

confidence_lower = mean_difference - margin_of_error
confidence_upper = mean_difference + margin_of_error

print("\n--- 95% CONFIDENCE INTERVAL ---")

print(
    "95% CI:",
    round(confidence_lower, 2),
    "to",
    round(confidence_upper, 2),
    "goals"
)


# ============================================================
# 8. STATISTICAL DECISION
# ============================================================

significance_level = 0.05

print("\n--- STATISTICAL DECISION ---")

if p_value < significance_level:
    print("Reject H0.")
    print(
        "There is a statistically significant difference "
        "between winning and losing teams."
    )
else:
    print("Fail to reject H0.")
    print(
        "There is insufficient evidence of a statistically "
        "significant difference."
    )


# ============================================================
# 9. FINAL CONCLUSION
# ============================================================

print("\n--- FINAL CONCLUSION ---")

print(
    f"Winning teams scored an average of "
    f"{winner_mean:.2f} goals per match."
)

print(
    f"Losing teams scored an average of "
    f"{loser_mean:.2f} goals per match."
)

print(
    f"The average difference was "
    f"{mean_difference:.2f} goals per match."
)

print(
    f"The paired t-test result was "
    f"t({degrees_of_freedom}) = "
    f"{t_statistic:.2f}, p < 0.001."
)

print(
    f"The 95% confidence interval was "
    f"{confidence_lower:.2f} to "
    f"{confidence_upper:.2f} goals."
)

print(
    "\nConclusion: There is strong statistical evidence "
    "that winning teams scored significantly more goals "
    "per match than losing teams."
)


# ============================================================
# 10. VISUALISATION
# ============================================================

groups = [
    "Winning Teams",
    "Losing Teams"
]

averages = [
    winner_mean,
    loser_mean
]

plt.figure(figsize=(8, 5))

plt.bar(groups, averages)

plt.title(
    "Average Goals Scored by Winning and Losing Teams"
)

plt.xlabel("Match Result")
plt.ylabel("Average Goals per Match")

plt.tight_layout()

plt.savefig(
    "objective1_average_goals.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({
    "Team Result": [
        "Winning Teams",
        "Losing Teams"
    ],
    "Mean Goals": [
        winner_mean,
        loser_mean
    ],
    "Standard Deviation": [
        winner_sd,
        loser_sd
    ]
})

summary["Mean Goals"] = summary["Mean Goals"].round(2)
summary["Standard Deviation"] = summary[
    "Standard Deviation"
].round(2)

print("\n--- SUMMARY TABLE ---")
print(summary.to_string(index=False))
