
import json

def extract_notebook_content(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        source = "".join(cell['source'])
        print(source[:500] + "..." if len(source) > 500 else source)
        if cell['cell_type'] == 'code':
            # Check for outputs specifically p-values or stats if printed
            outputs = cell.get('outputs', [])
            for output in outputs:
                if 'text' in output:
                    text = "".join(output['text'])
                    if len(text) < 200: # Only print short outputs (stats)
                        print(f"Output: {text.strip()}")

extract_notebook_content('template.ipynb')
