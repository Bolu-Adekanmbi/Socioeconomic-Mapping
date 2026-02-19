# Socioeconomic Mapping from Multi-Modal Remote Sensing Data

A machine learning-based approach to predict socioeconomic indicators (median household income, poverty rate, population) in low-resource contexts using satellite imagery, nighttime lights, and OpenStreetMap features.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Objectives](#project-objectives)
- [Study Area](#study-area)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Extending This Work](#extending-this-work)
- [Results & Evaluation](#results--evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

This project demonstrates a scalable framework for **socioeconomic mapping** in areas where traditional survey data may be sparse or outdated. By combining:
- **Satellite imagery** (Sentinel-2 multispectral data)
- **Nighttime lights** (VIIRS DNB monthly composites)
- **Geospatial features** (OpenStreetMap infrastructure data)
- **Population density** (WorldPop high-resolution rasters)

We train machine learning models to predict key socioeconomic indicators at the Census Block Group level, validated against American Community Survey (ACS) ground truth data.

**Why this matters:** In many low-resource contexts globally, traditional census data is infrequent or unavailable. This approach enables policymakers, researchers, and NGOs to estimate socioeconomic conditions using freely available remote sensing data and open geospatial repositories.

---

## Key Features

✅ **Multi-modal data fusion**: Integrates satellite imagery, nighttime lights, population data, and OSM infrastructure  
✅ **Spatial feature engineering**: Calculates densities, proximities, and zonal statistics for block groups  
✅ **Machine learning pipeline**: Implements Linear Regression, Random Forest, and XGBoost models  
✅ **Spatial cross-validation**: Evaluates model generalization using spatially-aware splits  
✅ **Reproducible workflow**: Jupyter notebooks with step-by-step data collection, preprocessing, and modeling  
✅ **Google Earth Engine integration**: Automated download of VIIRS nighttime lights data  
✅ **Scalable to other regions**: Framework can be adapted to any geographic area with available data

---

## Project Objectives

1. **Predict median household income** at the Census Block Group level
2. **Estimate poverty rates** using remotely sensed proxies
3. **Assess population distribution** patterns
4. **Evaluate model performance** using standard regression metrics (MAE, RMSE, R², MAPE)
5. **Provide a reusable framework** for similar studies in other geographic contexts

---

## Study Area

**Primary Study Region:**  
- **Montgomery County, Virginia** (FIPS: 51121)

**Expanded Multi-County Analysis:**  
The project also includes neighboring counties to improve model robustness:
- Craig County (FIPS: 51045)
- Roanoke County (FIPS: 51161)
- Roanoke City (FIPS: 51770)
- Salem City (FIPS: 51775)
- Floyd County (FIPS: 51063)
- Pulaski County (FIPS: 51155)
- Giles County (FIPS: 51071)
- Radford City (FIPS: 51750)

**Geographic Scope:** Southwestern Virginia, USA  
**Spatial Resolution:** Census Block Group level (~600-3,000 people per unit)  
**Temporal Coverage:** 2022 data (ACS 5-year estimates, satellite imagery, nighttime lights)

---

## Data Sources

### 1. **Satellite Imagery**
- **Sentinel-2**: Cloud-free composite at ~10m resolution
  - Bands used: B4 (Red), B8 (NIR), B11 (SWIR)
  - Derived indices: NDVI, NDBI, NDWI
  - Source: Google Earth Engine

### 2. **Nighttime Lights**
- **VIIRS DNB (Day/Night Band)**: Monthly stray light-corrected composites
  - Variable: `avg_rad` (average radiance)
  - Resolution: ~500m
  - Source: NOAA via Google Earth Engine

### 3. **Population Data**
- **WorldPop**: 100m resolution population count raster for 2022
  - Source: WorldPop Global Project

### 4. **OpenStreetMap (OSM) Features**
- **Infrastructure**: Roads, schools, healthcare facilities, grocery stores
  - Extracted using `OSMnx` Python library
  - Features: density, proximity, count within block groups

### 5. **Ground Truth (Census Data)**
- **American Community Survey (ACS)**: 5-year estimates (2018-2022)
  - Variables: Median household income, total population, poverty count/rate
  - Spatial unit: Block Groups
  - Source: U.S. Census Bureau via `censusdata` Python library

---

## Project Structure

```
SocioeconomicMapping-Project/
│
├── data/                              # Raw and processed datasets
│   ├── aligned_data/                  # Preprocessed and CRS-aligned files
│   │   ├── block_groups_clean_4326.geojson
│   │   ├── master_feature_matrix.csv  # Final feature matrix (Montgomery)
│   │   ├── engineered_features_*.csv   # Intermediate feature sets
│   │   └── *_4326.geojson             # Aligned OSM data
│   ├── neighboring_counties_data/     # Multi-county expanded dataset
│   │   └── master_feature_matrix_neighboring_counties.csv
│   ├── GDP_per_capita_1990_2022/      # Auxiliary economic data
│   ├── ground_truth.csv               # ACS socioeconomic indicators
│   ├── montgomery_boundary.geojson
│   ├── Sentinel2_2022.tif             # Satellite imagery
│   ├── VIIRS_Montgomery_2022.tif      # Nighttime lights
│   ├── usa_pop_2022_CN_100m_R2025A_v1.tif  # Population raster
│   └── *.geojson / *.gpkg             # OSM features (roads, schools, etc.)
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── version2_neighboring_counties.ipynb  # Primary analysis notebook
│   ├── clean_notebook.ipynb           # Cleaned workflow
│   ├── exploratoryNotebook1.ipynb     # Initial EDA
│   └── environment_API_Setup.ipynb    # GEE authentication setup
│
├── src/                               # Source code (modular scripts)
│   ├── download.py                    # Data download utilities
│   ├── preprocessing.py               # Data cleaning & alignment
│   ├── features.py                    # Feature engineering
│   ├── models.py                      # ML model training & evaluation
│   ├── frontend.py                    # Visualization & dashboard
│   └── run_pipeline.py                # End-to-end pipeline orchestration
│
├── version1_scripts/                  # Legacy scripts from initial version
│   ├── data_collection.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
│
├── outputs/                           # Model outputs, plots, and reports
│
├── GIS-Exploration/                   # QGIS project files for validation
│   ├── Raster_validation.qgz
│   └── GDP_Per_Capita_1990_2022_Analysis.qgz
│
├── pipelineVisualization/             # Workflow diagrams
│
├── environment.yml                    # Conda environment specification
├── README.md                          # This file
└── LICENSE                            # Apache 2.0 License

```

### Folder Descriptions

- **`data/`**: All raw and processed datasets. The `aligned_data/` subfolder contains CRS-standardized files (EPSG:4326) ready for analysis. `neighboring_counties_data/` extends the study to 9 counties.

- **`notebooks/`**: Interactive Jupyter notebooks for the complete workflow:
  - Data collection (Census API, OSM, Google Earth Engine)
  - Feature engineering (spatial proximity, density, zonal statistics)
  - Model training and evaluation
  - Visualizations and exploratory data analysis

- **`src/`**: Modular Python scripts (currently placeholders for future refactoring). Intended to contain production-ready code for pipeline automation.

- **`outputs/`**: Generated visualizations, trained model artifacts, and evaluation reports.

- **`GIS-Exploration/`**: QGIS project files for visual validation of raster data and spatial patterns.

---

## Methodology

### 1. **Data Collection**
- Download Census Block Group boundaries and ACS socioeconomic data via `censusdata`
- Extract OpenStreetMap features (roads, schools, healthcare, grocery) using `OSMnx`
- Retrieve satellite imagery and nighttime lights from Google Earth Engine
- Load WorldPop population raster

### 2. **Preprocessing**
- Reproject all geospatial layers to EPSG:4326 (WGS84)
- Clip rasters to study area boundaries
- Clean and filter OSM data (remove invalid geometries, duplicates)

### 3. **Feature Engineering**
For each Census Block Group, compute:

**A. OpenStreetMap-derived features:**
- Road density (km/km²)
- Distance to nearest school, healthcare facility, grocery store
- Count of amenities within 1km, 5km, 10km buffers

**B. Satellite-derived features:**
- Mean/median VIIRS nighttime light intensity
- Sentinel-2 spectral indices: NDVI, NDBI, NDWI
- Standard deviation of pixel values (heterogeneity proxy)

**C. Population features:**
- Total population count from WorldPop raster
- Population density

### 4. **Model Training**
- **Models tested**: Linear Regression, Random Forest, XGBoost
- **Target variables**: Median household income, poverty rate, population
- **Validation strategy**: 
  - 80/20 random train-test split
  - 5-fold spatial cross-validation (to avoid spatial autocorrelation bias)
- **Metrics**: MAE, RMSE, R², MAPE

### 5. **Evaluation**
- Compare model performance across random and spatial CV splits
- Analyze feature importance for tree-based models
- Generate residual plots and spatial error maps

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- Conda or virtualenv
- Google Earth Engine account (for satellite data download)
- Sufficient disk space (~5GB for rasters)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Bolu-Adekanmbi/Socioeconomic-Mapping.git
cd Socioeconomic-Mapping
```

### Step 2: Create Conda Environment
```bash
conda env create -f environment.yml
conda activate socioeconomic-mapping
```

**Alternatively, install packages manually:**
```bash
pip install geopandas pandas numpy matplotlib seaborn scikit-learn xgboost
pip install osmnx censusdata rasterio rioxarray rasterstats
pip install earthengine-api geemap geedim
```

### Step 3: Authenticate Google Earth Engine
Run the following in Python:
```python
import ee
ee.Authenticate()
ee.Initialize(project='your-gee-project-id')
```

### Step 4: Verify Installation
Open `notebooks/version2_neighboring_counties.ipynb` and run the first few cells to ensure all dependencies are loaded.

---

## Usage

### Running the Complete Workflow

**Option 1: Jupyter Notebook (Recommended for exploration)**
```bash
jupyter notebook notebooks/version2_neighboring_counties.ipynb
```
Follow the notebook cells sequentially:
1. Configure study area and data paths
2. Download Census boundaries and ACS data
3. Fetch OSM features
4. Download satellite imagery from GEE
5. Engineer features
6. Train and evaluate models

**Option 2: Automated Pipeline (Future implementation)**
```bash
python src/run_pipeline.py --config config.yaml
```
*(Note: `src/` scripts are currently placeholders. See [Extending This Work](#extending-this-work) for refactoring guidance.)*

### Quick Start Example

```python
import geopandas as gpd
import pandas as pd

# Load preprocessed feature matrix
features = pd.read_csv("data/aligned_data/master_feature_matrix.csv")

# Load ground truth
ground_truth = pd.read_csv("data/ground_truth.csv")

# Merge datasets
data = features.merge(ground_truth, on="GEOID")

# Train a simple model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = data.drop(columns=["GEOID", "median_income"])
y = data["median_income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

---

## Extending This Work

### To Apply This Framework to a New Region:

1. **Update geographic parameters** in the notebook:
   ```python
   STATE_FIPS = "06"  # California
   COUNTY_FIPS = ["001", "013"]  # Alameda, Contra Costa
   ```

2. **Adjust data paths**:
   - Modify raster file names for new Sentinel-2 and VIIRS downloads
   - Update Census API calls for the new state/county

3. **Re-run data collection cells**:
   - Census boundaries and ACS data
   - OSM feature extraction
   - GEE satellite downloads

4. **Feature engineering**:
   - The spatial functions (distance, density calculations) are scale-agnostic
   - Consider regional OSM data quality (some areas may have sparse coverage)

5. **Model retraining**:
   - Use the same ML pipeline
   - Optionally tune hyperparameters for the new dataset

### To Add New Features:

- **Additional OSM amenities**: 
  ```python
  parks = ox.features_from_polygon(region, tags={"leisure": "park"})
  ```
- **New satellite indices**: Modify Sentinel-2 band combinations
- **Temporal data**: Incorporate time-series nighttime lights for trend analysis

### To Modularize the Codebase:

Currently, the workflow is notebook-based. To convert to production scripts:

1. **Extract functions** from notebook cells into `src/` modules:
   - `download.py`: Data fetching (Census, OSM, GEE)
   - `preprocessing.py`: CRS alignment, clipping, cleaning
   - `features.py`: Spatial feature engineering logic
   - `models.py`: Model training, evaluation, cross-validation

2. **Create a configuration file** (`config.yaml`):
   ```yaml
   study_area:
     state_fips: "51"
     county_fips: ["121", "045", "161"]
   
   data_sources:
     sentinel2: "data/Sentinel2_2022.tif"
     viirs: "data/VIIRS_Montgomery_2022.tif"
     population: "data/usa_pop_2022_CN_100m_R2025A_v1.tif"
   
   models:
     - linear_regression
     - random_forest
     - xgboost
   ```

3. **Implement CLI interface** in `run_pipeline.py`:
   ```python
   import argparse
   import yaml
   
   parser = argparse.ArgumentParser()
   parser.add_argument("--config", required=True)
   args = parser.parse_args()
   
   config = yaml.safe_load(open(args.config))
   # Execute pipeline steps...
   ```

---

## Results & Evaluation

### Preliminary Findings (Montgomery County)

| Model              | MAE ($)  | RMSE ($) | R²    | MAPE (%) |
|--------------------|----------|----------|-------|----------|
| Linear Regression  | 16,327   | 19,825   | 0.196 | 27.2     |
| Random Forest      | 25,624   | 32,465   | -1.16 | 39.4     |
| XGBoost            | *TBD*    | *TBD*    | *TBD* | *TBD*    |

**Key Observations:**
- Linear models perform moderately well but underfit complex spatial patterns
- Tree-based models show high variance (likely due to limited training data)
- Spatial cross-validation reveals challenges in generalizing to distant areas
- Nighttime lights are the strongest predictor of income (feature importance analysis)

**Next Steps:**
- Expand training set to multi-county dataset (in progress)
- Incorporate additional features (e.g., land cover, road network centrality)
- Experiment with deep learning approaches (CNNs on satellite imagery patches)

---

## Contributing

Contributions are welcome! Potential areas for improvement:

- **Feature addition**: New satellite indices, OSM categories, or temporal features
- **Model enhancement**: Deep learning, ensemble methods, or causal inference
- **Code optimization**: Parallelize raster processing, optimize memory usage
- **Documentation**: Tutorials, API documentation, or case studies
- **Testing**: Unit tests for feature engineering and preprocessing functions

### Workflow:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/my-new-feature`)
5. Submit a Pull Request

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for the full license text.

### Commercial Use / Contact Clause

This code (and any substantial portion thereof) is provided under the Apache 2.0 license for general use, research, non-commercial projects, modifications, and distribution, **subject to the terms of that license**.

If you would like to use this work (or a substantial portion thereof) in a **commercial product** (i.e., a product or service that is charged for or intended for profit), please contact the author to discuss a separate commercial license.

The author retains all rights not expressly granted by the Apache 2.0 license.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{adekanmbi2025socioeconomic,
  author = {Adekanmbi, Mobolurin},
  title = {Socioeconomic Mapping from Multi-Modal Remote Sensing Data},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Bolu-Adekanmbi/Socioeconomic-Mapping}}
}
```

**Acknowledgment Example:**
> "Based in part on work by Mobolurin Adekanmbi (see [Socioeconomic-Mapping Repository](https://github.com/Bolu-Adekanmbi/Socioeconomic-Mapping))."

---

## Contact

For questions, collaboration opportunities, or commercial licensing inquiries:

- **Author**: Mobolurin Adekanmbi
- **GitHub**: [@Bolu-Adekanmbi](https://github.com/Bolu-Adekanmbi)
- **Project Repository**: [SocioeconomicMapping-Project](https://github.com/Bolu-Adekanmbi/Socioeconomic-Mapping)

---

