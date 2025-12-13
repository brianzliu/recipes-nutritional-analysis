import json

nb_path = 'template.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source = "".join(cell['source'])
    
    # Update Step 5: Predictor Variables
    if "Step 5: Framing a Prediction Problem" in source and "**Predictor Variables**" in source:
        # Check if we need to update
        if "calories_per_minute" not in source:
             new_source = source.replace(
                "total fat (PDV), sugar (PDV), protein (PDV), in addition to n_steps and calories (#).",
                "total fat (PDV), sugar (PDV), protein (PDV), calories_per_minute, complexity_density, in addition to n_steps and calories (#)."
            )
             cell['source'] = new_source.splitlines(keepends=True)
             print("Updated Step 5 variables.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
