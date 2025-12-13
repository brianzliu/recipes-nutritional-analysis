import json
import os

notebook_path = '/Users/brianliu/Documents/school/dsc80-2025-fa/projects/proj04/template.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Define new cells
new_cells = [
   {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
     "### Hyperparameter Tuning and Final Model\n",
     "We will use a **RandomForestRegressor** as our final model. \n",
     "We chose this model because it can capture non-linear relationships and interactions between features better than a simple linear model.\n",
     "\n",
     "**Features Added:**\n",
     "- `minutes`: Preparation time is a key factor in recipe complexity.\n",
     "- `n_ingredients`: More ingredients usually imply more complexity.\n",
     "- `total fat (PDV)`, `sugar (PDV)`, `protein (PDV)`: Nutritional components that might correlate with \"unhealthy\" or \"tasty\" ratings.\n",
     "\n",
     "**Preprocessing:**\n",
     "- **QuantileTransformer**: Applied to `minutes` and `calories (#)` to handle skewness and outliers.\n",
     "- **StandardScaler**: Applied to other numerical features to normalize their range.\n",
     "\n",
     "**Hyperparameters to Tune:**\n",
     "- `n_estimators`: Number of trees in the forest. We will try [50, 100].\n",
     "- `max_depth`: Maximum depth of the tree. We will try [5, 10, 15] to control overfitting.\n",
     "- `min_samples_split`: Minimum number of samples required to split an internal node. We will try [2, 5]."
    ]
   },
   {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
     "from sklearn.ensemble import RandomForestRegressor\n",
     "from sklearn.model_selection import GridSearchCV\n",
     "from sklearn.preprocessing import QuantileTransformer\n",
     "\n",
     "# 1. Define X and y with new features\n",
     "features = ['n_steps', 'calories (#)', 'minutes', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)']\n",
     "X_final = ri_custom_valid_rating[features]\n",
     "y_final = ri_custom_valid_rating['rating_avg']\n",
     "\n",
     "# 2. Train-test split (same random_state to ensure same split as baseline)\n",
     "X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.25, random_state=42)\n",
     "\n",
     "# 3. Preprocessing pipeline\n",
     "preprocessing_final = ColumnTransformer(\n",
     "    transformers=[\n",
     "        ('quantile', QuantileTransformer(output_distribution='normal'), ['minutes', 'calories (#)']),\n",
     "        ('scaling', StandardScaler(), ['n_steps', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)'])\n",
     "    ],\n",
     "    remainder='drop'\n",
     ")\n",
     "\n",
     "final_pipeline = Pipeline([\n",
     "    ('preprocessing', preprocessing_final),\n",
     "    ('regression', RandomForestRegressor(random_state=42))\n",
     "])\n",
     "\n",
     "# 4. Hyperparameter Tuning\n",
     "param_grid = {\n",
     "    'regression__n_estimators': [50, 100],\n",
     "    'regression__max_depth': [5, 10, 15],\n",
     "    'regression__min_samples_split': [2, 5]\n",
     "}\n",
     "\n",
     "grid_search = GridSearchCV(final_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)\n",
     "grid_search.fit(X_train_final, y_train_final)\n",
     "\n",
     "# 5. Best Model and Evaluation\n",
     "best_model = grid_search.best_estimator_\n",
     "\n",
     "print(\"Best Hyperparameters:\", grid_search.best_params_)\n",
     "\n",
     "y_pred_train_final = best_model.predict(X_train_final)\n",
     "y_pred_test_final = best_model.predict(X_test_final)\n",
     "\n",
     "rmse_train_final = np.sqrt(mean_squared_error(y_train_final, y_pred_train_final))\n",
     "rmse_test_final = np.sqrt(mean_squared_error(y_test_final, y_pred_test_final))\n",
     "\n",
     "print(f'Final Model Train RMSE: {rmse_train_final}')\n",
     "print(f'Final Model Test RMSE: {rmse_test_final}')\n",
     "\n",
     "# Comparison\n",
     "# Assuming rmse_test from baseline is available as variable 'rmse_test'\n",
     "# If not, we can't print the difference, but the values are printed above.\n",
     "try:\n",
     "    print(f\"Improvement over Baseline (Test RMSE): {rmse_test - rmse_test_final}\")\n",
     "except NameError:\n",
     "    print(\"Baseline RMSE variable not found for comparison.\")"
    ]
   }
]

# Find Step 7 location
step7_idx = -1
for i, cell in enumerate(nb['cells']):
    if "source" in cell and len(cell["source"]) > 0:
        source_text = "".join(cell["source"])
        if "## Step 7: Final Model" in source_text:
            step7_idx = i
            break

if step7_idx != -1:
    # Check if next cell is the placeholder TODO cell
    if step7_idx + 1 < len(nb['cells']):
        next_cell = nb['cells'][step7_idx + 1]
        source_text = "".join(next_cell.get("source", []))
        if "# TODO" in source_text:
            # Replace the TODO cell
            nb['cells'].pop(step7_idx + 1)
            # Insert new cells
            for cell in reversed(new_cells):
                nb['cells'].insert(step7_idx + 1, cell)
            print("Replaced TODO cell with Final Model code.")
        else:
            # Just insert after Step 7 markdown
            for cell in reversed(new_cells):
                nb['cells'].insert(step7_idx + 1, cell)
            print("Inserted Final Model code after Step 7 markdown.")
    else:
         # Append if Step 7 is last
        nb['cells'].extend(new_cells)
        print("Appended Final Model code to end.")
else:
    print("Could not find 'Step 7: Final Model'. Appending to end.")
    nb['cells'].extend(new_cells)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
