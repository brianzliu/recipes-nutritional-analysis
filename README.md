# Are Complicated Recipes Unhealthy and Boring?

**Name**: Brian Liu

## Introduction
The dataset used in this project is the **Recipes and Ratings** dataset, which nominally contains 83,782 recipes and 731,927 interactions (reviews and ratings) from Food.com. This project focuses on understanding how recipe complexity relates to nutritional value and user ratings.

**Research Question**: *How does the complexity of the recipe (measured by the number of steps) affect its caloric content and average user rating?*

**Significance**: This question explores the trade-off between effort and reward in cooking. Are complicated recipes "healthier" or "tastier"? Understanding this relationship can help home cooks decide if the extra effort is worth it and help recipe platforms recommend recipes that balance effort and nutrition.

**Relevant Columns**:
- `n_steps`: The number of steps required to make the recipe. (Proxy for complexity).
- `minutes`: The time it takes to prepare the recipe.
- `rating_avg`: The average rating (1-5 stars) given by users.
- `nutrition`: A list of nutritional values (calories, fat, sugar, etc.).
- `calories`: The number of calories per serving (extracted from `nutrition`).
- `sugar`: The sugar content (PDV) (extracted from `nutrition`).
- `protein`: The protein content (PDV) (extracted from `nutrition`).

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
To prepare the data for analysis, we performed the following cleaning steps:

1.  **Merging Datasets**: We left-merged the `recipes` dataset (containing recipe metadata) with the `interactions` dataset (containing user reviews) on `recipe_id`. This allows us to link recipe features (like steps) with user feedback (ratings).
2.  **Handling Zero Ratings**: We replaced `rating` values of `0` with `NaN`. In this dataset, a rating of 0 indicates that a user did not provide a numeric rating (often just a comment), not that they gave it a "zero star" review (the scale is 1-5). Treating 0 as a valid low score would severely skew the average ratings downwards incorrectly.
3.  **Calculating Average Ratings**: We grouped the merged data by recipe `id` and calculated the mean of the `rating` column, storing it as `rating_avg`. This gives us a single target variable per recipe.
4.  **Parsing Nutrition Data**: The `nutrition` column contained string representations of lists (e.g., `"[51.5, 0.0, ...]" `). We parsed these strings into actual lists and expanded them into separate columns: `calories`, `total fat`, `sugar`, `sodium`, `protein`, `saturated fat`, and `carbohydrates`. This allows us to analyze specific nutritional components directly.

**Cleaned Data Head**:
Here are the first 5 rows of the cleaned dataset (selected columns):

| name                                 |   n_steps |   calories |   rating_avg |
|:-------------------------------------|----------:|-----------:|-------------:|
| 1 brownies in the world    best ever |        10 |      138.4 |            4 |
| 1 in canada chocolate chip cookies   |        12 |      595.1 |            5 |
| 412 broccoli casserole               |         6 |      194.8 |            5 |
| millionaire pound cake               |         7 |      878.3 |            5 |
| 2000 meatloaf                        |        17 |      267   |            5 |

### Univariate Analysis
<iframe src="assets/univariate_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: The distribution of average ratings is left-skewed, with most recipes receiving high ratings (4-5 stars). This is typical for user-generated content where people tend to rate things they like.*

<iframe src="assets/univariate_2.html" width="800" height="600" frameborder="0"></iframe>
*Description: The distribution of the number of steps is right-skewed, with a median around 10 steps. Most recipes are relatively simple, but there are some extremely complex ones with over 50 steps.*


### Bivariate Analysis
<iframe src="assets/bivariate_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: There is a positive correlation between the number of steps and calories. More complex recipes tend to have more calories, although there is significant variance.*

<iframe src="assets/bivariate_2.html" width="800" height="600" frameborder="0"></iframe>
*Description: The relationship between saturated fat and rating is not very strong, but there is a slight trend where recipes with moderate saturated fat are rated highly. Extremely high fat content doesn't necessarily guarantee a better rating.*

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
*Description: This plot shows average ratings for recipes with different step counts. While very specific high-step counts have high averages (likely due to small sample sizes), the general trend is relatively flat.*

#### Average Recipe Calories, Sugar, and Total Fat per Recipe Complexity (Based on # of Steps)
| complexity_level (n_steps)  |   calories (#) |   sugar (PDV) |   total fat (PDV) |
|:-------------------|---------------:|--------------:|------------------:|
| Low (1-5)          |        315.745 |       59.7003 |           23.3981 |
| Medium (6-10)      |        397.962 |       59.3847 |           30.0082 |
| High (11-20)       |        480.02  |       66.5914 |           37.0473 |
| Very High (20+)    |        654.781 |       95.4448 |           51.343  |


## Assessment of Missingness
### NMAR Analysis
We suspect that the `rating` column might be NMAR (Not Missing At Random). People are often less likely to leave a rating when they feel indifferent about the recipe. If they loved it or hated it, they are more likely to rate it. However, if the missingness depends on the *unobserved* rating itself (e.g. low ratings are missing), it is NMAR. Additional data, such as "user engagement time" or "did they cook it", could help explain missingness and potentially make it MAR.

### Missingness Dependency
We tested if the missingness of `rating` depends on the number of steps (`n_steps`).

<iframe src="assets/missingness_nsteps.html" width="800" height="600" frameborder="0"></iframe>

**Results**:
- **Observed Statistic**: 1.339
- **P-value**: 0.0
- **Conclusion**: The p-value is < 0.05, so we reject the null hypothesis. The missingness of ratings **depends** on the number of steps in the recipe.

We also tested dependency on `minutes`.
- **Observed Statistic**: 51.45
- **P-value**: 0.138
- **Conclusion**: The p-value > 0.05, so we fail to reject the null hypothesis. We do not have evidence that rating missingness depends on the preparation time (`minutes`).

## Hypothesis Testing
**Null Hypothesis**: Recipes with 10+ steps and recipes with <10 steps have the same average calories.
**Alternative Hypothesis**: Recipes with 10+ steps have higher average calories.
**Test Statistic**: Difference in means (High Steps - Low Steps).

**Results**:
- **Observed Difference**: 128.1 calories
- **P-value**: 0.0
- **Conclusion**: Since the p-value is 0.0 (less than 0.05), we reject the null hypothesis. There is strong evidence that more complex recipes (10+ steps) have higher average calories than simpler recipes.

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
- **Train RMSE**: 0.4975
- **Test RMSE**: 0.4973

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
- **Best Hyperparameters**: `max_depth=15`, `min_samples_split=2`, `n_estimators=100`
- **Final Test RMSE**: 0.4543
- **Improvement**: Reduced RMSE by approx 0.043 compared to baseline.

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
- **RMSE (Short)**: 0.4202
- **RMSE (Long)**: 0.4738
- **Observed Absolute Difference**: 0.0536
- **P-value**: 0.0
- **Conclusion**: The p-value is 0.0, so we reject the null hypothesis. The model appears to be unfair, with significantly higher error (RMSE) for long recipes compared to short recipes.
