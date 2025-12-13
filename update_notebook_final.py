import json

nb_path = 'template.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source = "".join(cell['source'])
    
    # 1. Update Step 7 markdown with new features description
    if "### Hyperparameter Tuning and Final Model" in source:
        if "- `n_ingredients`" in source and "- `calories_per_minute`" not in source:
            new_source = source.replace(
                "- `nutrition` features (fat, sugar, protein) (Quantitative): To capture the \"health\" aspect. Standardized.",
                "- `nutrition` features (fat, sugar, protein) (Quantitative): To capture the \"health\" aspect. Standardized.\n- `calories_per_minute` (Transformed): `calories` / (`minutes` + 1). Captures energy density over time.\n- `complexity_density` (Transformed): `n_steps` / (`minutes` + 1). Captures the intensity of the cooking process (steps per minute)."
            )
            cell['source'] = new_source.splitlines(keepends=True)
            print("Updated Step 7 markdown.")

    # 2. Update the code cell for Final Model to include feature engineering
    if cell['cell_type'] == 'code' and "RandomForestRegressor" in source:
        # Check if we already added it
        if "'calories_per_minute'" not in source:
            # We need to prepend the feature engineering
            new_code = []
            lines = cell['source']
            for line in lines:
                if "X_final =" in line or "features =" in line:
                    # Inject before this
                    new_code.append("# Feature Engineering\n")
                    new_code.append("ri_custom_valid_rating['calories_per_minute'] = ri_custom_valid_rating['calories (#)'] / (ri_custom_valid_rating['minutes'] + 1)\n")
                    new_code.append("ri_custom_valid_rating['complexity_density'] = ri_custom_valid_rating['n_steps'] / (ri_custom_valid_rating['minutes'] + 1)\n")
                    new_code.append("\n")
                    break
            
            # Now rewrite the cell slightly to include them in features list and Preprocessor
            full_source = "".join(lines)
            full_source = full_source.replace(
                "features = ['n_steps', 'calories (#)', 'minutes', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)']",
                "features = ['n_steps', 'calories (#)', 'minutes', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)', 'calories_per_minute', 'complexity_density']"
            )
            full_source = full_source.replace(
                "('quantile', QuantileTransformer(output_distribution='normal'), ['minutes', 'calories (#)']),",
                "('quantile', QuantileTransformer(output_distribution='normal'), ['minutes', 'calories (#)', 'calories_per_minute']),"
            )
            full_source = full_source.replace(
                "('scaling', StandardScaler(), ['n_steps', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)'])",
                "('scaling', StandardScaler(), ['n_steps', 'n_ingredients', 'total fat (PDV)', 'sugar (PDV)', 'protein (PDV)', 'complexity_density'])"
            )
            
            # If we didn't inject above (because X_final line wasn't found exactly as iterated), ensure we inject at top of cell
            if "ri_custom_valid_rating['calories_per_minute']" not in full_source:
                 full_source = "# Feature Engineering\nri_custom_valid_rating['calories_per_minute'] = ri_custom_valid_rating['calories (#)'] / (ri_custom_valid_rating['minutes'] + 1)\nri_custom_valid_rating['complexity_density'] = ri_custom_valid_rating['n_steps'] / (ri_custom_valid_rating['minutes'] + 1)\n\n" + full_source

            cell['source'] = full_source.splitlines(keepends=True)
            print("Updated Step 7 code cell.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
