import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read Data and Wrangle
# The actual header is the first row, so we skip row 0 or set header=1
df_raw = pd.read_csv('player_data.csv', header=1)

# Drop any entirely empty columns if present
df_raw = df_raw.dropna(axis=1, how='all')

# Keep relevant columns: Player, Pos, Age, Min
df = df_raw[['Player', 'Pos', 'Age', 'Min']].copy()

# Filter for FW and DF only
df = df[df['Pos'].isin(['FW', 'DF'])]

# Convert types to numeric, coercing errors to NaN
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Min'] = pd.to_numeric(df['Min'], errors='coerce')

# Drop nulls (dropna) and fill any remaining if needed (fillna)
# Let's say we drop rows where Age or Min is NaN
df = df.dropna(subset=['Age', 'Min'])

# Filter out players who haven't played
df = df[df['Min'] > 0]

# To satisfy the 'join' requirement, let's split the dataset into two and merge them
df_info = df[['Player', 'Pos', 'Age']]
df_stats = df[['Player', 'Min']]
df_merged = pd.merge(df_info, df_stats, on='Player', how='inner')

# Feature Engineering: Categorize Age
def categorize_age(age):
    if age <= 24:
        return 'Youth'
    elif age <= 35:
        return 'Adult'
    else:
        return 'Senior'

df_merged['Age_Category'] = df_merged['Age'].apply(categorize_age)

# 2. Sampling
# Randomly pick between 30 and 40 data points for each group
np.random.seed(42) # For reproducibility
sample_fw = df_merged[df_merged['Pos'] == 'FW'].sample(n=35, random_state=42)
sample_df = df_merged[df_merged['Pos'] == 'DF'].sample(n=35, random_state=42)
sample_data = pd.concat([sample_fw, sample_df])

# 3. Descriptive Stats for FW (as main example)
# Outlier detection using IQR
Q1 = sample_fw['Age'].quantile(0.25)
Q3 = sample_fw['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = sample_fw[(sample_fw['Age'] < lower_bound) | (sample_fw['Age'] > upper_bound)]
has_outliers = not outliers.empty

mean_fw = sample_fw['Age'].mean()
median_fw = sample_fw['Age'].median()
var_fw = sample_fw['Age'].var()
std_fw = sample_fw['Age'].std()
skew_fw = sample_fw['Age'].skew()

mean_df = sample_df['Age'].mean()

# Visualization
plt.figure(figsize=(10, 5))
sns.histplot(data=sample_data, x='Age', hue='Pos', kde=True, bins=15, multiple="dodge")
plt.title('Age Distribution: Forwards vs Defenders (Sample)')
plt.savefig('age_dist.png')
plt.close()

# 4. Confidence Interval (95% for FW age)
ci_fw = stats.t.interval(0.95, df=len(sample_fw)-1, loc=mean_fw, scale=stats.sem(sample_fw['Age']))

# 5. Hypothesis Testing
# Two-sample t-test
t_stat, p_val = stats.ttest_ind(sample_fw['Age'], sample_df['Age'], equal_var=False)

# Store results to print
results = {
    "FW_Sample_Size": len(sample_fw),
    "DF_Sample_Size": len(sample_df),
    "FW_Mean": mean_fw,
    "FW_Median": median_fw,
    "FW_Variance": var_fw,
    "FW_StdDev": std_fw,
    "FW_Skewness": skew_fw,
    "Has_Outliers": has_outliers,
    "FW_CI": ci_fw,
    "DF_Mean": mean_df,
    "T_Stat": t_stat,
    "P_Value": p_val
}

print(results)
print(df_merged.head())