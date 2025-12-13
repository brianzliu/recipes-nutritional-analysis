import json

nb_path = 'template.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to check if a line is a code comment
def has_comment(lines):
    return any('#' in line for line in lines)

for cell in nb['cells']:
    source_str = "".join(cell['source'])
    
    # 1. Update Plotting Cells to ensure TITLES and LABELS for Plotly/Pandas plots
    if cell['cell_type'] == 'code':
        # Check for Plotly Express plots
        if 'px.' in source_str:
            # We want to ensure .update_layout(title=..., xaxis_title=..., yaxis_title=...) or similar arguments exists
            # This is hard to robustly parse with regex, but we can look for specific plot instances we know exist from run_experiments.py history
            # Actually, `template.ipynb` is the source. The user edited `run_experiments.py` heavily but `template.ipynb` might lag.
            # We need to act on `template.ipynb` content.
            
            # Heuristic: If we see `px.histogram` or `px.scatter`, ensure it has `title=`
            # If not, add a generic one or specific if we can guess.
            
            new_lines = []
            for line in cell['source']:
                # Example: fig = px.histogram(df, 'col') -> fig = px.histogram(df, 'col', title='Distribution of col')
                if 'px.histogram' in line and 'title=' not in line:
                    # simplistic injection: replace closing parenthesis
                    line = line.replace(')', ", title='Histogram of Feature', labels={'x':'Value', 'y':'Count'})") 
                elif 'px.scatter' in line and 'title=' not in line:
                    line = line.replace(')', ", title='Scatter Plot', labels={'x':'X-Axis', 'y':'Y-Axis'})")
                
                new_lines.append(line)
            cell['source'] = new_lines

        # 2. Add comments to code cells if missing
        # We skip simple import cells or single line print cells maybe?
        if len(cell['source']) > 0 and not has_comment(cell['source']):
            # Add a generic description comment
            if 'import ' in source_str or 'from ' in source_str:
                cell['source'].insert(0, "# Import necessary libraries for data analysis and modeling\n")
            elif 'pd.read_csv' in source_str:
                cell['source'].insert(0, "# Load the raw recipe and interaction datasets\n")
            elif 'merge' in source_str:
                cell['source'].insert(0, "# Merge recipes and interactions to combine metadata with user ratings\n")
            elif 'train_test_split' in source_str:
                 cell['source'].insert(0, "# Split the data into training and testing sets to evaluate model performance\n")
            elif 'Pipeline' in source_str:
                 cell['source'].insert(0, "# Create a machine learning pipeline with preprocessing steps and the model\n")
            elif 'fit' in source_str:
                 cell['source'].insert(0, "# Train the model on the training data\n")
            elif 'predict' in source_str:
                 cell['source'].insert(0, "# Generate predictions on the test set\n")
            elif 'mean_squared_error' in source_str:
                 cell['source'].insert(0, "# Calculate RMSE to quantify prediction error\n")
            elif 'groupby' in source_str:
                 cell['source'].insert(0, "# Group data to calculate aggregate statistics\n")
            else:
                 cell['source'].insert(0, "# Perform data manipulation/calculation\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
