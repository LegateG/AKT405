# Exploratory Data Analysis: Fallout: New Vegas Armor and Clothing Values

This project examines the in-game armor and clothing items from *Fallout: New Vegas* to determine how their attributes influence their in-game value (in caps, $). The dataset is sourced from the [Fallout: New Vegas Fandom Wiki](https://fallout.fandom.com/wiki/Fallout:_New_Vegas_armor_and_clothing#Armor). This EDA was originally a final project for my undergraduate statistics course in 2012 using IBM SPSS, now modernized for public sharing on GitHub.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Data Cleaning](#data-cleaning)
- [Descriptive Statistics](#descriptive-statistics)
- [Assumption Checks](#assumption-checks)
- [Multiple Linear Regression](#multiple-linear-regression)
- [Residual Analysis](#residual-analysis)
- [Variable Selection](#variable-selection)
- [Conclusion](#conclusion)

## Dataset Overview

The dataset consists of 36 observations of armor and clothing items, with the following variables:

- **Dependent Variable (y):** `value` (in-game value in caps, $)
- **Independent Variables:**
  - `DT` (Damage Threshold, numeric)
  - `weight` (Weight in pounds, numeric)
  - `healthPoint` (Health Points, numeric)
  - `type` (Categorical: 1 = Light Armor, 2 = Medium Armor, 3 = Heavy Armor/Power Armor)

### Sample Data

| DT | Weight | Health Point | Type | Value ($) |
|----|--------|--------------|------|-----------|
| 6  | 7      | 25           | 1    | 100       |
| 8  | 15     | 200          | 1    | 220       |
| ...| ...    | ...          | ...  | ...       |
| 12 | 30     | 100          | 3    | 1100      |
| 25 | 40     | 400          | 3    | 6500      |

Full data is provided in the raw data file (to be uploaded separately).

## Data Cleaning

### Outlier Removal

Initial residual analysis identified observations 4 and 16 as outliers based on Studentized residuals (> ±2) and Cook’s Distance (> 4/n = 0.1111). These were removed to improve model reliability.

- **Observation 4:** DT=14, Weight=20, HealthPoint=500, Type=1, Value=7500
- **Observation 16:** DT=22, Weight=23, HealthPoint=750, Type=2, Value=12500

## Descriptive Statistics

### Summary Statistics

```python
import pandas as pd
df = pd.read_csv("fallout_armor.csv")
print(df.describe())
```

| Variable       | Mean   | Std. Dev. | Min | Max  |
|---------------|--------|----------|-----|------|
| Value ($)     | 3,260.12 | 3,209.15 | 60  | 8,494 |
| DT           | 16.59  | 5.86     | 4   | 25   |
| Weight       | 25.65  | 11.80    | 5   | 45   |
| Health Point | 472.06 | 492.54   | 15  | 3,000 |
| Type         | 2.00   | 0.82     | 1   | 3    |

**Observation:** `value` and `healthPoint` show high variability, suggesting potential skewness or outliers (addressed in cleaning).

### Distribution Visualizations

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['value'], kde=True)
plt.title("Distribution of Value ($)")
plt.show()

sns.boxplot(x='type', y='value', data=df)
plt.title("Value ($) by Armor Type")
plt.show()
```

**Insight:** Heavy armor (Type 3) tends to have higher values, with wider variability.

## Assumption Checks

### 1. Normality

**Method:** Q-Q plot and One-Sample Kolmogorov-Smirnov Test

```python
from scipy.stats import kstest, norm
ks_stat, p_value = kstest(df['value'], 'norm', args=(df['value'].mean(), df['value'].std()))
print(f"KS Statistic: {ks_stat}, p-value: {p_value}")
```

**Result:** KS Z = 1.172, p = 0.128 > 0.05 → H₀ accepted → Normality assumption holds.

### 2. Linearity

```python
for var in ['DT', 'weight', 'healthPoint']:
    sns.scatterplot(x=df[var], y=df['value'])
    plt.title(f"Value ($) vs {var}")
    plt.show()
```

## Multiple Linear Regression

### Initial Model

```python
import statsmodels.api as sm
X = sm.add_constant(df[['DT', 'weight', 'healthPoint', 'type']])
y = df['value']
model = sm.OLS(y, X).fit()
print(model.summary())
```

**Model Summary:**

- R² = 0.584 (58.4% of variance explained)
- Adjusted R² = 0.527
- F-statistic = 10.175, p < 0.001 → Model is significant

### Residual Analysis

```python
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
sns.scatterplot(x=range(len(cooks_d)), y=cooks_d)
plt.axhline(y=4/len(df), color='r', linestyle='--')
plt.title("Cook's Distance")
plt.show()
```

## Variable Selection

### Methods Applied

- **Forward Selection:** Adds DT only (R² = 0.533).
- **Backward Elimination:** Removes healthPoint, type, weight, retains DT (R² = 0.533).
- **Stepwise:** Selects DT (R² = 0.533).

**Final Model:**

```
value = -2640.438 + 375.658 * DT ± 2226.978
```

- R²: 0.533 (53.3% of variance explained)
- Significance: F = 36.547, p < 0.001

## Conclusion

- **Key Finding:** DT (Damage Threshold) is the primary driver of value in *Fallout: New Vegas* armor and clothing, explaining 53.3% of the variance.
- **Limitations:** Non-significant predictors (`weight`, `healthPoint`, `type`) and detected multicollinearity (VIF > 5 for some variables) suggest potential model refinement.
- **Next Steps:** Explore non-linear models or interaction terms to capture additional variance.

**Feel free to fork this repository and experiment with the data!**
