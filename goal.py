#'''What is the average save percentage for starting goalkeepers who played in at
#  least three matches during the FIFA World Cup 2026? 
# Exclude goalkeepers with fewer than three appearances.'''


import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the dataset (assuming it's saved as 'data.csv')
df = pd.read_csv('data.csv', skiprows=2)  # Skip the two metadata rows above the column names

print(df.columns)
print(df.head())
# Step 1: Filter rows where the position (Pos) is 'GK' 
# and matches played (MP) is greater than or equal to 3
goalkeepers_filtered = df[(df['Pos'] == 'GK') & (df['MP'] >= 3)].copy()

# Step 2: Ensure relevant columns are numeric for subsequent descriptive statistics
goalkeepers_filtered['MP'] = pd.to_numeric(goalkeepers_filtered['MP'])
goalkeepers_filtered['Save%'] = pd.to_numeric(goalkeepers_filtered['Save%'])

# Step 3: Select the desired columns for your analysis
data = goalkeepers_filtered[['Player', 'Squad', 'Age', 'MP', 'Starts', 'Saves', 'Save%']]

#removing any rows with missing values in the 'Save%' column
df1 = data['Save%'].dropna()

print(f"Total qualified goalkeepers in sample (n): {len(df1)}")

# Calculating descriptive statistics for Save%
print("Descriptive Statistics for Save%:")
print(df1.describe())

Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df1[(df1 < lower_bound) | (df1 > upper_bound)]

if not outliers.empty:
    print(f"Outliers detected! Using Median as primary central tendency.")
    has_outliers = True
else:
    print(f"No statistical outliers detected. Mean is reliable.")
    has_outliers = False





mean_val = df1.mean()
median_val = df1.median()
std_dev = df1.std()
iqr_val = IQR

print(f"Mean Save%: {mean_val:.2f}%")
print(f"Median Save%: {median_val:.2f}%")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"IQR: {iqr_val:.2f}")
print(f"Variance: {df1.var():.2f}")

# Selected the appropriate measure based on outliers 


n = len(df1)
mean = np.mean(df1)
std_err = stats.sem(df1)

# 95% Confidence Interval
confidence_level = 0.95
ci = stats.t.interval(confidence_level, n - 1, loc=mean_val, scale=std_err)

print(f"Sample Mean: {mean:.2f}%")
print(f"95% Confidence Interval: {ci}")



# Hypothesized historical population mean
mu_historical = 70.0

# Perform One-Sample t-Test
t_stat, p_value = stats.ttest_1samp(df1, mu_historical)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Reject the null hypothesis (Statistically significant difference from past World Cups).")
else:
    print("Result: Fail to reject the null hypothesis (No significant difference from past World Cups).")