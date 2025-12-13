import json

nb_path = 'template.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper function to add a comment if not present
def add_comment(source_list, comment):
    if not source_list: return
    first_line = source_list[0].lstrip()
    if not first_line.startswith('#'):
        source_list.insert(0, f"# {comment}\n")

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Match content to context and add specific comments
        if "pd.read_csv" in source:
            add_comment(cell['source'], "Load dataframes for recipes and interactions")
            
        elif "recipes.merge(interactions" in source:
            add_comment(cell['source'], "Merge datasets and calculate average ratings per recipe to create a unified view")
            
        elif "ast.literal_eval" in source:
            add_comment(cell['source'], "Expand the stringified 'nutrition' list into separate columns for detailed analysis")
            
        elif "px.histogram(ri_custom, 'rating_avg'" in source:
             add_comment(cell['source'], "Visualize the distribution of average ratings to understand user satisfaction trends")
             
        elif "px.histogram(ri_custom, 'n_steps'" in source:
             add_comment(cell['source'], "Visualize the distribution of step counts to assess recipe complexity")
             
        elif "px.scatter" in source and "'n_steps', y='calories (#)'" in source:
            add_comment(cell['source'], "Bivariate Analysis: Explore relationship between steps and calories (using log scale for visibility)")
            
        elif "px.scatter" in source and "saturated fat" in source:
             add_comment(cell['source'], "Bivariate Analysis: Explore relationship between saturated fat and ratings (capping outliers)")
             
        elif "agg_rating_by_steps =" in source:
             add_comment(cell['source'], "Aggregate: Calculate average ratings for each step count (filtering for sufficient sample size)")
             
        elif "px.scatter(agg_rating_by_steps" in source:
             add_comment(cell['source'], "Visualize the aggregated trend of ratings vs steps with an OLS regression line")
             
        elif "pivot_table" in source:
             add_comment(cell['source'], "Create a pivot table to compare nutrition metrics across different complexity levels")
             
        elif "mean_missing = " in source:
             add_comment(cell['source'], "Calculate observed statistics for Missingness Dependency test")
             
        elif "permutation(shuffled_missing_pooled)" in source:
            add_comment(cell['source'], "Run permutation test to simulate the null distribution for missingness dependency")
            
        elif "LinearRegression" in source and "fit" in source:
            add_comment(cell['source'], "Train and evaluate the Baseline Linear Regression model using simple features")
            
        elif "RandomForestRegressor" in source and "GridSearchCV" in source:
             add_comment(cell['source'], "Define, tune, and train the Final Random Forest model with engineered features")
             
        elif "calculate_rmse" in source and "shuffled_labels" in source:
             add_comment(cell['source'], "Fairness Analysis: Permutation test to check if model error (RMSE) differs significantly between groups")
        
        # General fallbacks
        elif "train_test_split" in source and "LinearRegression" not in source and "RandomForest" not in source:
             add_comment(cell['source'], "Split data into training and testing sets")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
