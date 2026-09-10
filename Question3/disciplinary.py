import numpy as np
import pandas as pd
from scipy import stats

#Load raw dataset
df = pd.read_csv("match.csv", skiprows=1)

#Clean column headers and filter non-participants
df = df[df["Min"] > 0].copy()

#Create key derived metrics
df["CrdTotal"] = df["CrdY"] + df["CrdR"]
df["Cards_Per_90"] = np.where(df["90s"] > 0, df["CrdTotal"] / df["90s"], 0)

# Define Age Groups (Younger vs. Veteran)
# Population median age cutoff
median_age = df["Age"].median()
df["Age_Group"] = np.where(df["Age"] >= 28, "Veteran (28+)", "Younger (<28)")

# Display clean dataset preview
print(df[["Player", "Pos", "Age", "Min", "90s", "CrdY", "CrdR", "Cards_Per_90", "Age_Group"]].head())

# Stratified Sampling
np.random.seed(42)  # For reproducibility
sample_young = df[df["Age_Group"] == "Younger (<28)"].sample(n=30, replace=False)
sample_vet = df[df["Age_Group"] == "Veteran (28+)"].sample(n=30, replace=False)

sample_df = pd.concat([sample_young, sample_vet])

# Compute summary statistics by Age Group
desc_stats = sample_df.groupby("Age_Group")["Cards_Per_90"].agg(
    Mean='mean',
    Std_Dev='std',
    Median='median',
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
    Count='count'
)

print(desc_stats)

# 95% Confidence Interval for the combined sample mean
sample_data = sample_df["Cards_Per_90"]
n = len(sample_data)
mean = np.mean(sample_data)
sem = stats.sem(sample_data)

ci_95 = stats.t.interval(confidence=0.95, df=n-1, loc=mean, scale=sem)
print(f"95% Confidence Interval for Mean Cards Per 90: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")

# Conduct Two-Sample Independent t-Test (Welch's t-test assuming unequal variance)
t_stat, p_val = stats.ttest_ind(
    sample_young["Cards_Per_90"], 
    sample_vet["Cards_Per_90"], 
    equal_var=False
)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")