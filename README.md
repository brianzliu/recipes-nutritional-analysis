# Are Complicated Recipes Unhealthy and Boring?

**Name**: Brian Liu

## Introduction
The dataset used in this project is the **Recipes and Ratings** dataset, which contains information about recipes (minutes to prepare, number of steps, nutrition, etc.) and their user ratings.

**Research Question**: How does the complexity of the recipe (i.e. # of steps) affect the nutritional value and recipe rating?

This question is significant because it explores the trade-off between effort (complexity) and reward (nutrition/taste). Are complicated recipes worth the effort?

**Dataset Statistics**:
- **Rows**: 234,429 (before cleaning)
- **Columns**: `name`, `id`, `minutes`, `n_steps`, `nutrition`, `rating`, `n_ingredients`, etc.

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
1.  Merged `recipes` and `interactions` datasets.
2.  Filled missing ratings (0) with NaN.
3.  Calculated average rating per recipe (`rating_avg`).
4.  Expanded the `nutrition` column (list of values) into separate columns: `calories`, `total fat`, `sugar`, `sodium`, `protein`, `saturated fat`, `carbohydrates`.

### Univariate Analysis
<iframe src="assets/univariate_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: [Insert description of your univariate plot logic from notebook]*

<iframe src="assets/univariate_2.html" width="800" height="600" frameborder="0"></iframe>
*Description: [Insert description of your univariate plot logic from notebook]*


### Bivariate Analysis
<iframe src="assets/bivariate_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: [Insert description of your bivariate plot logic from notebook]*

<iframe src="assets/bivariate_2.html" width="800" height="600" frameborder="0"></iframe>
*Description: [Insert description of your bivariate plot logic from notebook]*

### Interesting Aggregates
#### Overall Average Rating and Recipe Count, Grouped By Number of Steps in Recipe
|   n_steps |   rating_avg |   count |
|----------:|-------------:|--------:|
|        39 |      4.93333 |      15 |
|        33 |      4.92308 |      41 |
|        40 |      4.89583 |      16 |
|        28 |      4.88991 |     157 |
|        50 |      4.84615 |      13 |
<iframe src="assets/interesting_aggregates_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: Aggregated data by complexity levels to see average nutritional values.*

#### Average Recipe Calories, Sugar, and Total Fat per Recipe Complexity (Based on # of Steps)
| complexity_level (n_steps)  |   calories (#) |   sugar (PDV) |   total fat (PDV) |
|:-------------------|---------------:|--------------:|------------------:|
| Low (1-5)          |        314.382 |       60.1612 |           23.2877 |
| Medium (6-10)      |        400.095 |       59.9385 |           30.1355 |
| High (11-20)       |        480.959 |       67.4381 |           36.8632 |
| Very High (20+)    |        637.803 |       87.1659 |           50.491  |


## Assessment of Missingness
### NMAR Analysis
We suspect that the `rating` column might be NMAR (Not Missing At Random) if people are less likely to leave a rating when they have a strong negative experience (or perhaps indifference), but this dependency isn't fully captured by observed variables.

### Missingness Dependency
We tested if the missingness of `rating` depends on the number of steps (`n_steps`).

<iframe src="assets/missingness_plot.html" width="800" height="600" frameborder="0"></iframe>

**Results**:
- **Observed Statistic**: [Insert Observed Stat from Notebook Step 3]
- **P-value**: [Insert P-value from Notebook Step 3]
- **Conclusion**: [Insert Conclusion, e.g., "The p-value is < 0.05, suggesting missingness depends on n_steps."]

## Hypothesis Testing
**Null Hypothesis**: Recipes with 10+ steps and recipes with <10 steps have the same average calories.
**Alternative Hypothesis**: Recipes with 10+ steps have higher average calories.
**Test Statistic**: Difference in means (High Steps - Low Steps).

**Results**:
- **P-value**: [Insert P-value from Notebook Step 4]
- **Conclusion**: [Insert Conclusion from Notebook Step 4]

## Framing a Prediction Problem
**Problem**: Predict the average rating of a recipe (`rating_avg`).
**Type**: Regression.
**Metric**: RMSE (Root Mean Squared Error).

We want to predict how well-rated a recipe will be based on its complexity and nutritional content. We use RMSE because it penalizes large errors and is in the same units as the rating (stars).

## Baseline Model
**Model**: Linear Regression
**Features**:
- `n_steps` (Quantitative)
- `calories (#)` (Quantitative)
**Preprocessing**: Standard Scaling.

**Performance**:
- **Train RMSE**: [Insert Train RMSE from Step 6]
- **Test RMSE**: [Insert Test RMSE from Step 6]

## Final Model
**Model**: Random Forest Regressor
**Features Added**:
- `minutes` (Quantile Transformed)
- `n_ingredients` (Standard Scaled)
- `total fat (PDV)` (Standard Scaled)
- `sugar (PDV)` (Standard Scaled)
- `protein (PDV)` (Standard Scaled)

**Hyperparameter Tuning**: Tuned `n_estimators`, `max_depth`, and `min_samples_split` using GridSearchCV.

**Performance**:
- **Best Hyperparameters**: [Insert Best Params from Step 7]
- **Final Test RMSE**: [Insert Final RMSE from Step 7]

## Fairness Analysis
**Question**: Does the model perform differently for Short recipes (< 30 min) vs. Long recipes (>= 30 min)?
**Group X**: Short recipes
**Group Y**: Long recipes
**Metric**: RMSE

**Hypotheses**:
- **Null**: Model RMSE is roughly the same for both groups.
- **Alternative**: Model RMSE differs significantly.

<iframe src="assets/fairness_plot.html" width="800" height="600" frameborder="0"></iframe>

**Results**:
- **Observed Absolute Difference**: [Insert Observed Diff from Step 8]
- **P-value**: [Insert P-value from Step 8]
- **Conclusion**: [Insert Conclusion from Step 8]
