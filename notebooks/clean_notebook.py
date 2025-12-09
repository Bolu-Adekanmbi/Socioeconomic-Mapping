import json
import sys

# Load the notebook
with open('Notebook1Copy.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Total cells: {len(notebook['cells'])}")
print(f"Code cells: {sum(1 for c in notebook['cells'] if c['cell_type']=='code')}")
print(f"Markdown cells: {sum(1 for c in notebook['cells'] if c['cell_type']=='markdown')}")

# Find cells with reasoning/errors
reasoning_cells = []
error_cells = []
visualization_cells = []

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        text = ' '.join(cell.get('source', [])).lower()
        # Check for reasoning/failure/issue keywords
        if any(keyword in text for keyword in ['reasoning', '**reasoning**', 'failed', 'error', 'issue', 'try again', 'previous', 'attempt']):
            reasoning_cells.append(i)
    
    if cell['cell_type'] == 'code':
        # Check for errors in outputs
        has_error = False
        has_viz = False
        
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error' or output.get('name') == 'stderr':
                has_error = True
            # Check for visualizations (display_data with image)
            if output.get('output_type') == 'display_data' and 'image/png' in output.get('data', {}):
                has_viz = True
        
        if has_error:
            error_cells.append(i)
        if has_viz:
            visualization_cells.append(i)

print(f"\nReasoning/failure markdown cells: {len(reasoning_cells)}")
if reasoning_cells:
    print(f"First 20 indices: {reasoning_cells[:20]}")

print(f"\nCells with errors: {len(error_cells)}")
if error_cells:
    print(f"First 20 indices: {error_cells[:20]}")

print(f"\nVisualization cells (must keep): {len(visualization_cells)}")
if visualization_cells:
    print(f"First 20 indices: {visualization_cells[:20]}")

# Create cleaned notebook
cleaned_cells = []

# Add setup section at beginning
setup_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Socioeconomic Mapping using Satellite Imagery and Geographic Data\n",
        "\n",
        "## Prerequisites and Setup\n",
        "\n",
        "### Required Software and Libraries\n",
        "- Python 3.8 or higher\n",
        "- Jupyter Notebook or Google Colab\n",
        "\n",
        "### Required Python Packages\n",
        "```bash\n",
        "pip install geopandas pandas numpy matplotlib seaborn\n",
        "pip install scikit-learn xgboost\n",
        "pip install earthengine-api\n",
        "pip install osmnx\n",
        "pip install rasterio shapely\n",
        "pip install requests tqdm\n",
        "```\n",
        "\n",
        "### Google Earth Engine Setup\n",
        "1. Create a Google Earth Engine account at https://earthengine.google.com/\n",
        "2. Create a cloud project and note your project ID\n",
        "3. Authenticate using:\n",
        "```python\n",
        "import ee\n",
        "ee.Authenticate()\n",
        "ee.Initialize(project='your-project-id')\n",
        "```\n",
        "\n",
        "### API Keys and Authentication\n",
        "- Google Earth Engine project ID (set as environment variable or in code)\n",
        "- Census API key (optional but recommended): https://api.census.gov/data/key_signup.html\n",
        "\n",
        "### Data Sources\n",
        "This project uses:\n",
        "- **U.S. Census Bureau**: American Community Survey (ACS) data, TIGER/Line shapefiles\n",
        "- **Google Earth Engine**: Sentinel-2 satellite imagery, VIIRS nighttime lights\n",
        "- **OpenStreetMap**: Infrastructure and facility data via OSMnx\n",
        "\n",
        "---\n"
    ]
}

cleaned_cells.append(setup_markdown)

# Track cells to skip (errors, reasoning)
skip_indices = set(reasoning_cells + error_cells)

# Keep cells that are not in skip list
for i, cell in enumerate(notebook['cells']):
    if i not in skip_indices:
        cleaned_cells.append(cell)
    elif i in visualization_cells:  # Keep visualizations even if they have errors
        cleaned_cells.append(cell)

print(f"\nCleaned notebook will have {len(cleaned_cells)} cells (original: {len(notebook['cells'])})")
print(f"Removed {len(notebook['cells']) - len(cleaned_cells)} cells")

# Create cleaned notebook
cleaned_notebook = {
    "cells": cleaned_cells,
    "metadata": notebook.get('metadata', {}),
    "nbformat": notebook.get('nbformat', 4),
    "nbformat_minor": notebook.get('nbformat_minor', 5)
}

# Save cleaned notebook
with open('Notebook1Copy_cleaned.ipynb', 'w', encoding='utf-8') as f:
    json.dump(cleaned_notebook, f, indent=2, ensure_ascii=False)

print("\nCleaned notebook saved as 'Notebook1Copy_cleaned.ipynb'")
