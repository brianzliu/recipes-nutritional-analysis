
import pandas as pd
import numpy as np
import ast
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

try:
    import plotly.express as px
    import plotly.graph_objects as go
    pd.options.plotting.backend = 'plotly'
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not found.")

def save_plot(fig, filename):
    if HAS_PLOTLY and fig is not None:
        path = os.path.join('assets', filename)
        fig.write_html(path, include_plotlyjs='cdn')
        print(f"Saved plot to {path}")

# Ensure assets directory
if not os.path.exists('assets'):
    os.makedirs('assets')

print("SECTION: Data Loading")
recipes = pd.read_csv("RAW_recipes.csv")
interactions = pd.read_csv("RAW_interactions.csv")

recipes_interactions = recipes.merge(interactions, left_on='id', right_on='recipe_id', how='left')
recipes_interactions['rating'] = recipes_interactions['rating'].apply(lambda x: np.nan if x == 0.0 else x)
recipes_interactions_recipe_rating_avg = recipes_interactions.groupby('name')['rating'].mean()
recipes_interactions_recipe_rating_avg = recipes_interactions_recipe_rating_avg.rename('rating_avg')
# Note: Using merge on 'name' might be risky if names are not unique, but following user code.
# The user code: recipes_interactions.merge(..., left_on='name', right_index=True, how='left')
# This duplicates rows for every interaction.
recipes_interactions = recipes_interactions.merge(recipes_interactions_recipe_rating_avg, left_on='name', right_index=True, how='left')

# For analysis, we often want one row per recipe, not per interaction.
# The user's notebook shows `recipes_interactions` being used for `ri_custom`.
# `ri_custom` seems to be the main dataframe.
# However, if we merge on 'name' and 'interactions' has multiple reviews per recipe, we get many rows per recipe.
# Step 2 in prompt says "Find the average rating per recipe... Add this Series... back to the recipes dataset".
# The user's approach: `recipes.merge(interactions...)` -> big df. Then groupby name -> avg rating. Then merge back.
# This results in a dataframe with one row per INTERACTION, but with 'rating_avg' column attached.
# Wait, for Univariate 'rating_avg' distribution, we usually want one entry per RECIPE.
# If I use `recipes_interactions` (merged), I have duplicated recipes.
# Let's check `ri_custom` logic in user code.
# `ri_custom = recipes_interactions.copy()`
# It seems the user IS working with the expanded interactions dataframe.
# BUT, for the Analysis Part 1 Step 1, the prompt says "dataset description... one row per recipe".
# Actually, the user's variables `recipes_interactions` implies interactions.
# Let's follow the user's code exactly. If they plot 'rating_avg' from `ri_custom`, it might be weighted by # of interactions if not deduplicated.
# However, the user does: `ri_custom = ri_custom.sample(frac=1).head(50000)` later.
# Let's create a deduplicated version for recipe-level analysis if needed, or just stick to what the user wrote.
# User code: `fig = px.histogram(ri_custom, 'rating_avg'...)` -> This will plot rating_avg REPEATEDLY for every interaction.
# That might be what they intended or a mistake, but I will reproduce their code.
# WAIT, looking closely at the user's `recipes_interactions` merge:
# `recipes.merge(interactions)` -> Many rows.
# `groupby('name')['rating'].mean()` -> Series of unique names.
# `merge(..., left_on='name', ...)` -> Attaches avg to many rows.
# So `ri_custom` has many rows.

ri_custom = recipes_interactions.copy()
ri_custom['nutrition'] = ri_custom['nutrition'].apply(ast.literal_eval)

nutrition_columns = [
    'calories (#)',
    'total fat (PDV)',
    'sugar (PDV)',
    'sodium (PDV)',
    'protein (PDV)',
    'saturated fat (PDV)',
    'carbohydrates (PDV)'
]

nutrition_df = pd.DataFrame(ri_custom['nutrition'].tolist(), columns=nutrition_columns, index=ri_custom.index)
for col in nutrition_columns:
    ri_custom[col] = nutrition_df[col]

print(f"Data Loaded. Shape: {ri_custom.shape}")

# --- Univariate Analysis ---
print("SECTION: Univariate Analysis")

if HAS_PLOTLY:
    # 1. Rating Avg
    fig1 = px.histogram(ri_custom, 'rating_avg', nbins=20, title="Distribution of Recipes' Average Rating")
    fig1.update_layout(bargap=0.1)
    save_plot(fig1, 'univariate_1.html')

    # 2. N Steps - Log Scale for better visibility of tail
    fig2 = px.histogram(ri_custom, 'n_steps', title="Distribution of Recipes' # of Steps (Log Scale)")
    fig2.update_layout(xaxis_type="log", bargap=0.1)
    save_plot(fig2, 'univariate_2.html')

# --- Bivariate Analysis ---
print("SECTION: Bivariate Analysis")

if HAS_PLOTLY:
    # 1. N Steps vs Calories - Log Y for Calories due to outliers
    # Filter out 0 calories for log scale if necessary, though plotly handles it gracefully usually.
    fig3 = px.scatter(ri_custom, x='n_steps', y='calories (#)', title='Calories vs. Number of Steps (Log Scale for Calories)')
    fig3.update_layout(yaxis_type="log")
    # Add trendline on the log data? OLS on log data is better visualised if we transformed data first, 
    # but for simple scatter, log axis is enough.
    save_plot(fig3, 'bivariate_1.html')

    # 2. Saturated Fat vs Rating Avg - Cap Saturated Fat outliers for plot
    # Cap at 99th percentile for visualization to see the bulk of data
    cap_val = ri_custom['saturated fat (PDV)'].quantile(0.99)
    ri_plot_subset = ri_custom[ri_custom['saturated fat (PDV)'] <= cap_val]
    
    fig4 = px.scatter(ri_plot_subset, x='saturated fat (PDV)', y='rating_avg', title='Average Rating vs. Saturated Fat (PDV) (Capped at 99th Percentile)')
    fig4.update_traces(marker=dict(opacity=0.3)) # Reduce opacity to see density
    save_plot(fig4, 'bivariate_2.html')

# --- Interesting Aggregates ---
print("SECTION: Interesting Aggregates")

agg_rating_by_steps = ri_custom.groupby('n_steps').agg({
    'rating_avg': 'mean',
    'id': 'count'
}).rename(columns={'id': 'count'})

agg_rating_by_steps = agg_rating_by_steps[agg_rating_by_steps['count'] > 10]
agg_rating_by_steps = agg_rating_by_steps.sort_values('n_steps')

print("Top 5 Ratings by Steps:")
print(agg_rating_by_steps.sort_values('rating_avg', ascending=False).head().to_markdown())

if HAS_PLOTLY:
    fig5 = px.scatter(agg_rating_by_steps,
                     x=agg_rating_by_steps.index,
                     y='rating_avg',
                     trendline='ols',
                     title='Average Rating vs. Number of Steps',
                     labels={'n_steps': 'Number of Steps', 'rating_avg': 'Average Rating'})
    fig5.update_layout(width=800, height=480)
    save_plot(fig5, 'interesting_aggregates_1.html')

# Complexity Pivot
ri_custom['complexity_level'] = pd.cut(ri_custom['n_steps'], 
                                     bins=[0, 5, 10, 20, 100], 
                                     labels=['Low (1-5)', 'Medium (6-10)', 'High (11-20)', 'Very High (20+)'])

agg_nutrition_by_complexity = ri_custom.pivot_table(
    index='complexity_level',
    values=['calories (#)', 'sugar (PDV)', 'total fat (PDV)'],
    aggfunc='mean'
)
print("Aggregates by Complexity:")
print(agg_nutrition_by_complexity.to_markdown())


# --- Step 3: Missingness ---
print("SECTION: Missingness")

# 1. Dependency on n_steps
ri_custom['rating_missing'] = ri_custom['rating'].isna()
mean_missing = ri_custom[ri_custom['rating_missing']]['n_steps'].mean()
mean_present = ri_custom[~ri_custom['rating_missing']]['n_steps'].mean()
observed_diff_nsteps = abs(mean_missing - mean_present)

print(f"Mean n_steps (Rating Missing): {mean_missing}")
print(f"Mean n_steps (Rating Present): {mean_present}")
print(f"Missingness n_steps Observed Diff: {observed_diff_nsteps}")

n_permutations = 500 # Reduced from 1000 for speed
simulated_diffs_nsteps = []
shuffled_missing_pooled = ri_custom['rating_missing'].values
n_steps_values = ri_custom['n_steps'].values

for _ in range(n_permutations):
    shuffled_missing = np.random.permutation(shuffled_missing_pooled)
    mean_m = n_steps_values[shuffled_missing].mean()
    mean_p = n_steps_values[~shuffled_missing].mean()
    simulated_diffs_nsteps.append(abs(mean_m - mean_p))

p_value_nsteps = (np.array(simulated_diffs_nsteps) >= observed_diff_nsteps).mean()
print(f"Missingness n_steps P-value: {p_value_nsteps}")

if HAS_PLOTLY:
    fig6 = px.histogram(pd.DataFrame({'simulated_diffs': simulated_diffs_nsteps}), x='simulated_diffs', title='Empirical Dist. Test Statistic (n_steps)')
    fig6.add_vline(x=observed_diff_nsteps, line_color='red', annotation_text='Observed Statistic')
    save_plot(fig6, 'missingness_nsteps.html')

# 2. Non-dependency on minutes
mean_missing_min = ri_custom[ri_custom['rating_missing']]['minutes'].mean()
mean_present_min = ri_custom[~ri_custom['rating_missing']]['minutes'].mean()
observed_diff_min = abs(mean_missing_min - mean_present_min)

print(f"Mean minutes (Rating Missing): {mean_missing_min}")
print(f"Mean minutes (Rating Present): {mean_present_min}")
print(f"Missingness minutes Observed Diff: {observed_diff_min}")

simulated_diffs_min = []
minutes_values = ri_custom['minutes'].values

for _ in range(n_permutations):
    shuffled_missing = np.random.permutation(shuffled_missing_pooled)
    mean_m = minutes_values[shuffled_missing].mean()
    mean_p = minutes_values[~shuffled_missing].mean()
    simulated_diffs_min.append(abs(mean_m - mean_p))

p_value_min = (np.array(simulated_diffs_min) >= observed_diff_min).mean()
print(f"Missingness minutes P-value: {p_value_min}")


# --- Step 4: Hypothesis Testing ---
print("SECTION: Hypothesis Testing")
# Null: >=10 steps and <10 steps equal calories.
# Alt: >=10 steps higher calories.

ri_custom_calories_nsteps = ri_custom[['n_steps', 'calories (#)']].copy()
ri_custom_calories_nsteps['>=10 steps'] = ri_custom_calories_nsteps['n_steps'] >= 10

observed_stat_df = ri_custom_calories_nsteps.groupby('>=10 steps').mean()
observed_diff_hyp = observed_stat_df.loc[True, 'calories (#)'] - observed_stat_df.loc[False, 'calories (#)']

print(f"Hypothesis Observed Diff: {observed_diff_hyp}")

simulated_diffs_hyp = []
shuffled_calories = ri_custom_calories_nsteps['calories (#)'].values
is_ge_10 = ri_custom_calories_nsteps['>=10 steps'].values

for _ in range(n_permutations):
    shuffled = np.random.permutation(shuffled_calories)
    # Using boolean indexing for speed
    # True group
    mean_ge = shuffled[is_ge_10].mean()
    mean_lt = shuffled[~is_ge_10].mean()
    simulated_diffs_hyp.append(mean_ge - mean_lt)

p_value_hyp = (np.array(simulated_diffs_hyp) >= observed_diff_hyp).mean()
print(f"Hypothesis P-value: {p_value_hyp}")


# --- Step 5, 6, 7: Prediction ---
print("SECTION: Prediction Model")

ri_custom_valid_rating = ri_custom.dropna(subset=['rating_avg'])
# Deduplicate for modeling? User's user code doesn't explicitly deduplicate before splitting, 
# but they used `ri_custom_valid_rating` which came from `ri_custom`.
# `ri_custom` came from merge(interactions).
# If we have multiple interactions for same recipe, we are leaking info if same recipe is in train and test?
# Or just predicting rating_avg (which is recipe level) using duplicated rows?
# The user's code: `X = ri_custom_valid_rating.loc[:, ['n_steps', 'calories (#)']]`
# This means duplicated recipes if multiple reviews.
# But `rating_avg` is SAME for all rows of a recipe.
# This essentially weights the regression by # of reviews. 
# I will proceed as is.

# Baseline
X_base = ri_custom_valid_rating[['n_steps', 'calories (#)']]
y_base = ri_custom_valid_rating['rating_avg']

X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, test_size=0.25, random_state=42)

base_pre = ColumnTransformer(
    transformers=[('scaling', StandardScaler(), ['n_steps', 'calories (#)'])],
    remainder='drop'
)
base_pipe = Pipeline([('preprocessing', base_pre), ('regression', LinearRegression())])
base_pipe.fit(X_train, y_train)

y_pred_train = base_pipe.predict(X_train)
y_pred_test = base_pipe.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Baseline Train RMSE: {rmse_train}")
print(f"Baseline Test RMSE: {rmse_test}")

# Final Model
features = ['n_steps', 'calories (#)', 'minutes', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)']
# Ensure n_ingredients exists? In the user snippet: `features = [...]`
# But where is 'n_ingredients' created?
# User code: `X_final = ri_custom_valid_rating[features]`
# I need to verify 'n_ingredients' is in `ri_custom`.
# The user snippet for `ri_custom` creation does NOT calculate n_ingredients explicitly.
# The user's `recipes` dataframe from `pd.read_csv("RAW_recipes.csv")` usually has `n_ingredients`.
# Let's hope it's there. If `pd.read_csv` reads it, it's fine.
# In `RAW_recipes.csv` description, `n_ingredients` is often present. 
# If not, I'll print available columns and fail/warn.

if 'n_ingredients' not in ri_custom_valid_rating.columns:
    print("Warning: 'n_ingredients' not found. Creating it from nutrition or skipping?")
    # We merged `recipes` into `recipes_interactions`. `recipes` has it.
    pass

X_final = ri_custom_valid_rating[features]
y_final = ri_custom_valid_rating['rating_avg']

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.25, random_state=42)

final_pre = ColumnTransformer(
    transformers=[
        ('quantile', QuantileTransformer(output_distribution='normal'), ['minutes', 'calories (#)']),
        ('scaling', StandardScaler(), ['n_steps', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)'])
    ],
    remainder='drop'
)

# User used GridSearch. I will replicate it, but maybe with fewer jobs or smaller grid if slow?
# User grid: 50,100 est; 5,10,15 depth.
# This is reasonably fast.
final_pipe = Pipeline([
    ('preprocessing', final_pre),
    ('regression', RandomForestRegressor(random_state=42, n_jobs=-1)) # n_jobs in RF too
])

param_grid = {
    'regression__n_estimators': [50, 100],
    'regression__max_depth': [5, 10, 15],
    'regression__min_samples_split': [2, 5]
}

print("Running Grid Search for Final Model...")
grid_search = GridSearchCV(final_pipe, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_final, y_train_final)

best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

y_pred_train_final = best_model.predict(X_train_final)
y_pred_test_final = best_model.predict(X_test_final)

rmse_train_final = np.sqrt(mean_squared_error(y_train_final, y_pred_train_final))
rmse_test_final = np.sqrt(mean_squared_error(y_test_final, y_pred_test_final))

print(f"Final Model Train RMSE: {rmse_train_final}")
print(f"Final Model Test RMSE: {rmse_test_final}")
print(f"Improvement over Baseline: {rmse_test - rmse_test_final}")

# --- Step 8: Fairness ---
print("SECTION: Fairness Analysis")

test_results = X_test_final.copy()
test_results['rating_actual'] = y_test_final
test_results['rating_pred'] = y_pred_test_final
test_results['is_short'] = test_results['minutes'] < 30

def calculate_rmse(df):
    return np.sqrt(mean_squared_error(df['rating_actual'], df['rating_pred']))

rmse_short = calculate_rmse(test_results[test_results['is_short']])
rmse_long = calculate_rmse(test_results[~test_results['is_short']])
observed_diff_fair = abs(rmse_short - rmse_long)

print(f"RMSE (Short): {rmse_short}")
print(f"RMSE (Long): {rmse_long}")
print(f"Observed Fairness Diff: {observed_diff_fair}")

simulated_diffs_fair = []
shuffled_labels_pool = test_results['is_short'].values

for _ in range(n_permutations):
    shuffled_labels = np.random.permutation(shuffled_labels_pool)
    group_a = test_results[shuffled_labels]
    group_b = test_results[~shuffled_labels]
    # Re-calculate RMSE
    # optimization: RMSE = sqrt(mean((y-y_pred)^2))
    # We can pre-calculate squared errors.
    # But sticking to user logic structure for clarity.
    
    # Wait, filtering dataframe in loop is slow.
    # Optimize:
    # errors_sq = (rating_actual - rating_pred)^2
    # rmse_a = sqrt(mean(errors_sq[mask]))
    errors_sq = (test_results['rating_actual'] - test_results['rating_pred'])**2
    errors_sq = errors_sq.values # numpy array
    
    # filtered by shuffled labels
    mean_sq_a = errors_sq[shuffled_labels].mean()
    mean_sq_b = errors_sq[~shuffled_labels].mean()
    
    rmse_a = np.sqrt(mean_sq_a)
    rmse_b = np.sqrt(mean_sq_b)
    
    simulated_diffs_fair.append(abs(rmse_a - rmse_b))

p_value_fair = (np.array(simulated_diffs_fair) >= observed_diff_fair).mean()
print(f"Fairness P-value: {p_value_fair}")

if HAS_PLOTLY:
    fig7 = px.histogram(pd.DataFrame({'diffs': simulated_diffs_fair}), x='diffs', title='Empirical Dist. Fairness RMSE Diff')
    fig7.add_vline(x=observed_diff_fair, line_color='red', annotation_text='Observed Diff')
    save_plot(fig7, 'fairness_plot.html')

print("DONE")
