import pandas as pd
import numpy as np
import ast
import dataframe_image as dfi

print("Loading data...")
recipes = pd.read_csv("RAW_recipes.csv")
interactions = pd.read_csv("RAW_interactions.csv")

recipes_interactions = recipes.merge(interactions, left_on='id', right_on='recipe_id', how='left')
recipes_interactions['rating'] = recipes_interactions['rating'].apply(lambda x: np.nan if x == 0.0 else x)
recipes_interactions_recipe_rating_avg = recipes_interactions.groupby('name')['rating'].mean()
recipes_interactions_recipe_rating_avg = recipes_interactions_recipe_rating_avg.rename('rating_avg')
recipes_interactions = recipes_interactions.merge(recipes_interactions_recipe_rating_avg, left_on='name', right_index=True, how='left')

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

print("Exporting image...")
# Select nice columns for the image, or just head(5) as requested
# Using exactly what user tried: ri_custom.head(5)
dfi.export(ri_custom.head(5), 'ri_custom_head.png', table_conversion='chrome')
print("Image exported to ri_custom_head.png")
