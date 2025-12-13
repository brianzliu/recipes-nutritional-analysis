import json

nb_path = 'template.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source = "".join(cell['source'])
    
    # 1. Update Step 5 with Final Model variables
    if "Step 5: Framing a Prediction Problem" in source and "**Predictor Variables**" in source:
        new_source = source.replace(
            "- Baseline Model: calories and n_steps. Since these variables are known before any user rates the recipe, they are valid variables to predict average rating.",
            "- Baseline Model: calories and n_steps. Since these variables are known before any user rates the recipe, they are valid variables to predict average rating.\n- Final Model: minutes, n_ingredients, total fat (PDV), sugar (PDV), protein (PDV), in addition to n_steps and calories (#)."
        )
        cell['source'] = new_source.splitlines(keepends=True)
        print("Updated Step 5 variables.")

    # 2. Update Step 7 with Hyperparameter explanations
    if "### Hyperparameter Tuning and Final Model" in source and "**Hyperparameters to Tune:**" in source:
        # We need to inject the "WHY" part
        new_lines = []
        for line in cell['source']:
            if "- `n_estimators`:" in line:
                new_lines.append("- `n_estimators`: Number of trees in the forest. We will try [50, 100]. **Why**: More trees usually give better performance but with diminishing returns and higher cost.\n")
            elif "- `max_depth`:" in line:
                new_lines.append("- `max_depth`: Maximum depth of the tree. We will try [5, 10, 15] to control overfitting. **Why**: Deeper trees capture more complex patterns but can memorize noise (overfit). Shallower trees generalize better but might underfit.\n")
            elif "- `min_samples_split`:" in line:
                new_lines.append("- `min_samples_split`: Minimum number of samples required to split an internal node. We will try [2, 5]. **Why**: Higher values prevent the model from learning distinct rules for very small groups of samples, further reducing overfitting.\n")
            else:
                new_lines.append(line)
        cell['source'] = new_lines
        print("Updated Step 7 hyperparameter explanations.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
