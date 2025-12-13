# Are Complicated Recipes Unhealthy and Boring?

**Name**: Brian Liu

## Introduction
The dataset used in this project is the **Recipes and Ratings** dataset, which nominally contains 83,782 recipes and 731,927 interactions (reviews and ratings) from Food.com. After preprocessing, there are 234,429 rows. This project focuses on understanding how recipe complexity relates to nutritional value and user ratings.

**Research Question**: *What is the relationship between a recipe's complexity (steps, time, ingredients) and its nutritional value, and can these factors be used to accurately predict user satisfaction (average ratings)?*

**Significance**: This analysis investigates the "return on investment" for home cooking. We explore whether "complex" recipes (more steps, longer time) are necessarily more calorie-dense or higher-rated than simple ones. By building a model to predict ratings from complexity and nutrition, and assessing its fairness across short and long recipes, we aim to understand what affects user satisfaction for recipes.

**Relevant Columns**:
- `n_steps`: The number of steps required to make the recipe (i.e. how complex a given recipe is).
- `minutes`: The time it takes to prepare the recipe.
- `rating_avg`: The average rating (1-5 stars) given by users for a given recipe.
- `nutrition`: A list of nutritional values (calories, fat, sugar, etc.), out of which calories, total fat, sugar, and protein columns were extracted.
- `calories (#)`: The number of calories per serving (extracted from `nutrition`).
- `sugar (PDV)`: The sugar content (PDV) (extracted from `nutrition`).
- `total fat (PDV)`: The total fat content (PDV) (extracted from `nutrition`).
- `protein (PDV)`: The protein content (PDV) (extracted from `nutrition`).

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
To prepare the data for analysis, we performed the following cleaning steps:

1.  We left-merged the `recipes` dataset (containing recipe metadata) with the `interactions` dataset (containing user reviews) on `recipe_id`. This allows us to link recipe features (like steps) with user feedback (ratings).
2.  We replaced `rating` values of `0` with `NaN`. In this dataset, a rating of 0 indicates that a user did not provide a numeric rating (often just a comment), not that they gave it a "zero star" review (the scale is 1-5). Treating 0 as a valid low score would severely skew the average ratings downwards incorrectly.
3.  We grouped the merged data by recipe `id` and calculated the mean of the `rating` column, storing it as `rating_avg`. This gives us a single target variable per recipe.
4.  The `nutrition` column contained string representations of lists (e.g., `"[51.5, 0.0, ...]" `). We parsed these strings into actual lists and expanded them into separate columns: `calories`, `total fat`, `sugar`, `sodium`, `protein`, `saturated fat`, and `carbohydrates`. This allows us to analyze specific nutritional components directly.

**Cleaned Data Head**:
| | name | id | minutes | contributor_id | submitted | tags | nutrition | n_steps | steps | description | ingredients | n_ingredients | user_id | recipe_id | date | rating | review | rating_avg |
|---:|:---|---:|---:|---:|:---|:---|:---|---:|:---|:---|:---|---:|---:|---:|:---|---:|:---|---:|
| 0 | 1 brownies in the world best ever | 333281 | 40 | 985201 | 2008-10-27 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0] | 10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil', 'combine chocolate and butter in a medium saucepan and cook over medium-low heat , stirring frequently , until evenly melted', 'remove from heat and let cool to room temperature', 'combine eggs , sugar , cocoa powder , vanilla extract , espresso , and salt in a large bowl and briefly stir until just evenly incorporated', 'add cooled chocolate and mix until uniform in color', 'add flour and stir until just incorporated', 'transfer batter to the prepared baking dish', 'bake until a tester inserted in the center of the brownies comes out clean , about 25 to 30 minutes', 'remove from the oven and cool completely before cutting'] | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven! | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour'] | 9 | 386585.0 | 333281.0 | 2008-11-19 | 4.0 | These were pretty good, but took forever to bake. I would send it ended up being almost an hour! Even then, the brownies stuck to the foil, and were on the overly moist side and not easy to cut. They did taste quite rich, though! Made for My 3 Chefs. | 4.0 |
| 1 | 1 in canada chocolate chip cookies | 453467 | 45 | 1848091 | 2011-04-11 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings'] | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] | 12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl , sift together the flours and baking powder', 'set aside', 'in another mixing bowl , blend together the sugars , margarine , and salt until light and fluffy', 'add the eggs , water , and vanilla to the margarine / sugar mixture and mix together until well combined', 'add in the flour mixture to the wet ingredients and blend until combined', 'scrape down the sides of the bowl and add the chocolate chips', 'mix until combined', 'scrape down the sides to the bowl again', 'using an ice cream scoop , scoop evenly rounded balls of dough and place of cookie sheet about 1 - 2 inches apart to allow for spreading during baking', 'bake for 10 - 15 minutes or until golden brown on the outside and soft & chewy in the center', 'serve hot and enjoy !'] | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead. | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips'] | 11 | 424680.0 | 453467.0 | 2012-01-26 | 5.0 | Originally I was gonna cut the recipe in half (just the 2 of us here), but then we had a park-wide yard sale, & I made the whole batch & used them as enticements for potential buyers ~ what the hey, a free cookie as delicious as these are, definitely works its magic! Will be making these again, for sure! Thanks for posting the recipe! | 5.0 |
| 2 | 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli'] | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0] | 6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce', 'pour into baking dish , sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly , about 10 more minutes'] | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions'] | 9 | 29782.0 | 306168.0 | 2008-12-31 | 5.0 | This was one of the best broccoli casseroles that I have ever made. I made my own chicken soup for this recipe. I was a bit worried about the tsp of soy sauce but it gave the casserole the best flavor. YUM! <br> The photos you took (shapeweaver) inspired me to make this recipe and it actually does look just like them when it comes out of the oven. <br> Thanks so much for sharing your recipe shapeweaver. It was wonderful! Going into my family's favorite Zaar cookbook :) | 5.0 |
| 3 | 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli'] | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0] | 6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce', 'pour into baking dish , sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly , about 10 more minutes'] | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions'] | 9 | 1196280.0 | 306168.0 | 2009-04-13 | 5.0 | I made this for my son's first birthday party this weekend. Our guests INHALED it! Everyone kept saying how delicious it was. I was I could have gotten to try it. | 5.0 |
| 4 | 412 broccoli casserole | 306168 | 40 | 50969 | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli'] | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0] | 6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce', 'pour into baking dish , sprinkle remaining cheese over top', 'bake for 25 minutes or until cheese is lightly browned', 'sprinkle with rest of french fried onions and bake until onions are browned and cheese is bubbly , about 10 more minutes'] | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions'] | 9 | 768828.0 | 306168.0 | 2013-08-02 | 5.0 | Loved this. Be sure to completely thaw the broccoli. I didn&#039;t and it didn&#039;t get done in time specified. Just cooked it a little longer though and it was perfect. Thanks Chef. | 5.0 |

### Univariate Analysis
<iframe src="assets/univariate_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: The distribution of average ratings is left-skewed, with most recipes receiving high ratings (4-5 stars).*
<iframe src="assets/bivariate_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: There is a positive correlation between the number of steps and calories. More complex recipes tend to have more calories, although there is significant variance.*

<iframe src="assets/bivariate_2.html" width="800" height="600" frameborder="0"></iframe>
*Description: The relationship between saturated fat and rating is not very strong, but there is a slight trend where recipes with moderate saturated fat are rated highly. Extremely high fat content doesn't necessarily guarantee a better rating.*

### Interesting Aggregates
#### Overall Average Rating and Recipe Count, Grouped By Number of Steps in Recipe

| n_steps | rating_avg | count |
|--------:|-----------:|------:|
|      39 |    4.93333 |    15 |
|      33 |    4.92308 |    41 |
|      40 |    4.89583 |    16 |
|      28 |    4.88991 |   157 |
|      50 |    4.84615 |    13 |

<iframe src="assets/interesting_aggregates_1.html" width="800" height="600" frameborder="0"></iframe>
*Description: This plot shows average ratings for recipes with different step counts. While very specific high-step counts have high averages (likely due to small sample sizes), the general trend is relatively flat.*

#### Average Recipe Calories, Sugar, and Total Fat per Recipe Complexity (Based on # of Steps)
This table reveals a clear trend: as recipe complexity increases (more steps), the nutritional content (calories, sugar, and fat) also increases. "Very High" complexity recipes have nearly double the calories of "Low" complexity ones. This suggests that more effort in the kitchen often correlates with richer, heavier meals (like elaborate desserts or main courses), whereas simpler recipes might be lighter snacks or quick meals.

| complexity_level (n_steps)  |   calories (#) |   sugar (PDV) |   total fat (PDV) |
|:-------------------|---------------:|--------------:|------------------:|
| Low (1-5)          |        315.745 |       59.7003 |           23.3981 |
| Medium (6-10)      |        397.962 |       59.3847 |           30.0082 |
| High (11-20)       |        480.02  |       66.5914 |           37.0473 |
| Very High (20+)    |        654.781 |       95.4448 |           51.343  |


## Assessment of Missingness
### NMAR Analysis
The `rating` column might be **NMAR** (Not Missing At Random). Users are often less likely to leave a rating when they feel indifferent about the recipe; they are motivated to rate only if they have a strong positive or negative experience. However, because this "indifference" is the unobserved value itself (or related to the unobserved rating), the missingness is NMAR. If we could collect additional data such as "user engagement time" or "did they verify cooking it", we might be able to explain the missingness, potentially making it MAR (Missing At Random) conditioned on those new columns.

### Missingness Dependency
We tested if the missingness of `rating` depends on the **number of steps** (`n_steps`).
- **Null Hypothesis**: The distribution of `n_steps` is the same when `rating` is missing vs. not missing.
- **Alternative Hypothesis**: The distribution of `n_steps` is different when `rating` is missing vs. not missing.
- **Test Statistic**: The absolute difference in means of `n_steps`.
- **Significance Level**: 0.05.

<iframe src="assets/missingness_nsteps.html" width="800" height="600" frameborder="0"></iframe>

**Results**:
- **Observed Statistic**: 1.34
- **P-value**: **< 0.001**
- **Conclusion**: Since the p-value is less than the significance level of 0.05, we **reject the null hypothesis**. There is strong evidence that the missingness of ratings depends on the `n_steps` column.

We also tested dependency on `minutes`.
- **P-value**: 0.108
- **Conclusion**: The p-value > 0.05, so we **fail to reject the null hypothesis**. We do not have sufficient evidence to say rating missingness depends on recipe preparation time.

## Hypothesis Testing
**Null Hypothesis**: Recipes with high complexity (10+ steps) and low complexity (<10 steps) have the same average caloric content.
**Alternative Hypothesis**: Recipes with high complexity (10+ steps) have a **higher** average caloric content than low complexity recipes. (One-sided).

**Test Statistic**: Difference in means (Mean Calories of High Complexity - Mean Calories of Low Complexity).
**Significance Level**: 0.05.

**Justification**: We chose difference in means because `calories` is a quantitative variable and we are comparing two groups. A permutation test is appropriate here as we are making no assumptions about the underlying distribution of calories.

**Results**:
- **Observed Difference**: 128.1 calories
- **P-value**: **< 0.001**
- **Conclusion**: Since the p-value is below 0.05, we **reject the null hypothesis**. The data suggests that more complex recipes indeed tend to be more calorie-dense.

## Framing a Prediction Problem
**Problem**: Predict the average rating of a recipe (`rating_avg`).
**Type**: **Regression**.
**Response Variable**: `rating_avg`. We chose this variable because we want to quantify user satisfaction on a continuous scale (1-5), and predicting the exact average allows for more granular recommendations.
**Evaluation Metric**: **RMSE** (Root Mean Squared Error). We chose RMSE over $R^2$ because RMSE provides an error metric in the same units as the rating (stars), which is more interpretable for this context. We want to know, on average, how many stars off our prediction is.

**Features Known at Prediction Time**: We only use features intrinsic to the recipe (steps, ingredients, nutrition, time) which are known *before* any users rate it. We do not use any user-interaction data (like number of reviews) as predictors.

## Baseline Model
**Model**: Linear Regression.
**Features**:
- `n_steps` (Quantitative): Used as is (standardized).
- `calories` (Quantitative): Used as is (standardized).

**Preprocessing**: We used a `StandardScaler` for both quantitative features to ensure they are on the same scale, which is standard practice for linear regression (though strictly not required for prediction performance, it helps with interpretation of coefficients). There were no categorical features in the baseline model.

**Performance**:
- **Train RMSE**: 0.4975
- **Test RMSE**: 0.4973
- **Assessment**: The model performs adequately but simply assumes a linear relationship between complexity/calories and rating, which may be too simple. The RMSE of ~0.5 means we are off by half a star on average.

## Final Model
**Model**: Random Forest Regressor.
**Features Added**:
- `minutes` (Quantitative): Recipes taking longer might be rated differently (e.g. "Sunday roasts" vs "quick snacks"). We applied a `QuantileTransformer` to handle the heavy right skew of time data.
- `n_ingredients` (Quantitative): Another proxy for complexity. Standardized.
- `nutrition` features (fat, sugar, protein) (Quantitative): To capture the "health" aspect. Standardized.

**Algorithm Choice**: A **Random Forest** model was chosen because it can capture non-linear relationships and interactions between features (e.g., highly complex recipes might only be rated highly if they are also high in fat/sugar).

**Hyperparameter Tuning**: We used `GridSearchCV` with 3-fold cross-validation. We tuned:
- `n_estimators` (Number of trees): [50, 100]. More trees generally improve stability and performance but increase computation.
- `max_depth` (Tree depth): [5, 10, 15]. Controlling depth helps prevent overfitting by limiting how specific the model can get to the training data.
- `min_samples_split` (Split criteria): [2, 5]. Higher values prevent the model from learning from too few samples (noise).

**Performance**:
- **Best Hyperparameters**: `max_depth=15`, `min_samples_split=2`, `n_estimators=100`.
- **Final Test RMSE**: 0.4542
- **Improvement**: The Final Model reduced the RMSE by approximately **0.043 stars** compared to the Baseline.
- **Why it Improved (Data Generating Process)**: The baseline model only looked at steps and calories linearly. However, in the real world (the data generating process), user satisfaction is complex. A recipe isn't rated lower just because it has more calories; in fact, high fat/sugar often makes food taste better (higher ratings), but extremely long prep times (`minutes`) might frustrate users (lower ratings). The Random Forest captures these non-linear interactions: it can learn that "high calories + short time = high rating" (tasty quick snack) but "high calories + very long time + complicated steps = lower rating" (too much effort). Adding `minutes` and nutritional details gave the model the necessary context to distinguish between "good" complexity and "bad" complexity.

## Fairness Analysis
**Question**: Does the model perform differently for **Short recipes** (< 30 min) vs. **Long recipes** (>= 30 min)?

**Group X**: Short recipes (`minutes` < 30).

**Group Y**: Long recipes (`minutes` >= 30).

**Evaluation Metric**: **RMSE**.

**Hypotheses**:
- **Null Hypothesis**: The model's RMSE is the same for both Short and Long recipes. (Any difference is due to chance).
- **Alternative Hypothesis**: The model's RMSE is different for Short and Long recipes. (Two-sided).
- **Significance Level**: 0.05.
- **Test Statistic**: Absolute Difference in RMSE.

<iframe src="assets/fairness_plot.html" width="800" height="600" frameborder="0"></iframe>

**Results**:
- **RMSE (Short)**: 0.4202
- **RMSE (Long)**: 0.4738
- **Observed Absolute Difference**: 0.0536
- **P-value**: **< 0.001**
- **Conclusion**: Since the p-value < 0.05, we **reject the null hypothesis**. The model is **unfair** with respect to recipe duration; it is significantly more accurate (lower RMSE) for short recipes than for long recipes. This might be because short recipes are simpler to rate or generally have less variance in quality.
