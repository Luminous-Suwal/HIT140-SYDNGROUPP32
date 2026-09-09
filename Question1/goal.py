#Question: What is the average save percentage for starting goalkeepers who played in at
# least three matches during the FIFA World Cup 2026? 
# Exclude goalkeepers with fewer than three appearances.

#importing libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv('data.csv', skiprows=2)  # Skipping the two metadata rows above the column names

print(df.columns)
print(df.head())

# Filtering rows where the position (Pos) is 'GK' 
# and matches played (MP) is greater than or equal to 3
goalkeepers_filtered = df[(df['Pos'] == 'GK') & (df['MP'] >= 3)].copy()

# Ensuring relevant columns are numeric for subsequent descriptive statistics
goalkeepers_filtered['MP'] = pd.to_numeric(goalkeepers_filtered['MP'])
goalkeepers_filtered['Save%'] = pd.to_numeric(goalkeepers_filtered['Save%'])

#Engineering new features based on the known data for deeper insight
goalkeepers_filtered['Saves_per_Match'] = goalkeepers_filtered['Saves'] / goalkeepers_filtered['MP']
goalkeepers_filtered['Estimated_Shots_Faced'] = np.round(goalkeepers_filtered['Saves'] / (goalkeepers_filtered['Save%'] / 100))


# Selelcting only relevant columns for your analysis
data = goalkeepers_filtered[['Player', 'Squad', 'Age', 'MP', 'Starts', 'Saves', 'Save%', 'Saves_per_Match','Estimated_Shots_Faced']]

print(data.head())
#cleaning up the data by dropping any rows that are missing a 'Save%' value.
df_cleaned = data['Save%'].dropna()

print(f"Total qualified goalkeepers in sample (n): {len(df_cleaned)}")

print(df_cleaned.head())

#get the sample of 30 with random function from python 
df1 = df_cleaned.sample(n=30, random_state=42)

# Calculating descriptive statistics for Save%
print("Descriptive Statistics for Save%:")
print(df1.describe())

#Checking for outliers using Interquartile Range (IQR) method on the sample
Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df1[(df1 < lower_bound) | (df1 > upper_bound)]

if not outliers.empty:
    print(f"Outliers detected! Using Median as the measure of center.")
    has_outliers = True
else:
    print(f"No statistical outliers detected. Using Mean as the measure of center.")
    has_outliers = False

mean_val = df1.mean()
median_val = df1.median()
std_dev = df1.std()
var = df1.var()
iqr_val = IQR

print(f"Mean Save%: {mean_val:.2f}%")
print(f"Median Save%: {median_val:.2f}%")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"IQR: {iqr_val:.2f}")
print(f"Variance: {var:.2f}")

#MEASURE DISTRIBUTION
#checking skewness and plotted a histogram to see if it's symmetric or asymmetric.
skewness = df1.skew()
print(f"Distribution Skewness: {skewness:.2f} (0 is perfectly symmetric)")

plt.hist(df1, bins=8, color='skyblue', edgecolor='black')
plt.title("Distribution of Goalkeeper Save Percentages (n=35)")
plt.xlabel("Save Percentage (%)")
plt.ylabel("Frequency")
plt.show()

#calculated the 95% confidence interval for the sample mean.
n = len(df1)
mean = np.mean(df1)
std_err = stats.sem(df1)
confidence_level = 0.95
ci = stats.t.interval(confidence_level, n - 1, loc=mean_val, scale=std_err)

print(f"Sample Mean: {mean:.2f}%")
print(f"95% Confidence Interval: {ci}")

# HYPOTHESIS TESTING (One-Sample t-Test)
#  doing a one-sample t-test because n > 30 (t-statistics still apply).
# Hypothesized historical population mean
mu_historical = 70.0
t_stat, p_value = stats.ttest_1samp(df1, mu_historical)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Reject the null hypothesis (Statistically significant difference from past World Cups).")
else:
    print("Result: Fail to reject the null hypothesis (No significant difference from past World Cups).")