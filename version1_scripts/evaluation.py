import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import copy # To create new model instances for retraining in sparsity simulation
import joblib # For loading models

# Assuming modeling.py is in the same directory and contains calculate_metrics
# If modeling.py needs to be self-contained for execution, calculate_metrics might need to be redefined or included.
# For modularity, we assume it's imported.

# Re-define calculate_metrics here to avoid circular imports if modeling.py depends on evaluation.py
# Or to ensure evaluation.py is standalone.
def calculate_metrics(y_true, y_pred):
    """
    Calculates common regression evaluation metrics.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: A dictionary containing MAE, RMSE, R-squared, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE calculation, handling potential division by zero
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R-squared': r2, 'MAPE': mape}

def plot_predictions_vs_actual(y_true, y_pred, title, ax):
    """
    Generates a scatter plot comparing actual vs. predicted values with a 45-degree reference line.

    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes): Axes object to plot on.
    """
    ax.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min()) if len(y_true) > 0 and len(y_pred) > 0 else 0
    max_val = max(y_true.max(), y_pred.max()) if len(y_true) > 0 and len(y_pred) > 0 else 1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2) # 45-degree line
    ax.set_xlabel('Actual Median Income ($)')
    ax.set_ylabel('Predicted Median Income ($)')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val * 0.9, max_val * 1.1)
    ax.set_ylim(min_val * 0.9, max_val * 1.1)

def categorize_income(income_value, threshold):
    """
    Categorizes an income value as 'low_income' or 'other_income' based on a threshold.

    Args:
        income_value (float): The income value to categorize.
        threshold (float): The low-income threshold.

    Returns:
        str: 'low_income' or 'other_income'.
    """
    if income_value <= threshold:
        return 'low_income'
    else:
        return 'other_income'

def evaluate_policy_metrics(
    y_true_all, y_pred_all, low_income_threshold,
    model_name="Model", plot_confusion_matrix=True
):
    """
    Evaluates policy-relevant metrics (precision, recall, confusion matrix) for low-income identification.

    Args:
        y_true_all (pd.Series): All actual median income values.
        y_pred_all (array-like): All predicted median income values.
        low_income_threshold (float): The threshold to classify low income.
        model_name (str): Name of the model for printing results.
        plot_confusion_matrix (bool): Whether to print the confusion matrix.

    Returns:
        tuple: Precision and Recall for the 'low_income' class.
    """
    y_true_categories = y_true_all.apply(lambda x: categorize_income(x, low_income_threshold))
    y_pred_categories = pd.Series(y_pred_all).apply(lambda x: categorize_income(x, low_income_threshold))

    precision = precision_score(y_true_categories, y_pred_categories, pos_label='low_income', zero_division=0)
    recall = recall_score(y_true_categories, y_pred_categories, pos_label='low_income', zero_division=0)

    print(f"\n{model_name}:")
    print(f"  Precision (low_income): {precision:.4f}")
    print(f"  Recall (low_income): {recall:.4f}")

    if plot_confusion_matrix:
        labels = sorted(y_true_categories.unique())
        cm = confusion_matrix(y_true_categories, y_pred_categories, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f'Actual {label}' for label in labels], columns=[f'Predicted {label}' for label in labels])
        print(f"\nConfusion Matrix for {model_name}:")
        print(cm_df)
    return precision, recall

def evaluate_urban_rural_performance(master_feature_matrix, y_actual, y_pred, model_prefix="model"):
    """
    Analyzes model performance (MAE, RMSE) in urban vs. rural areas.

    Args:
        master_feature_matrix (pd.DataFrame): DataFrame with 'population_density_per_sqkm' and 'GEOID'.
        y_actual (pd.Series): Actual median income values.
        y_pred (array-like): Predicted median income values.
        model_prefix (str): Prefix for model names in output (e.g., 'Tuned XGBoost').

    Returns:
        pd.DataFrame: DataFrame with urban/rural performance metrics.
    """
    population_density_threshold = master_feature_matrix['population_density_per_sqkm'].median()
    master_feature_matrix['urban_rural_class'] = master_feature_matrix['population_density_per_sqkm'].apply(
        lambda x: 'urban' if x > population_density_threshold else 'rural'
    )

    performance_df = pd.DataFrame({
        'actual': y_actual,
        'predicted': y_pred,
        'urban_rural_class': master_feature_matrix['urban_rural_class']
    })

    results = []
    for area_type in ['urban', 'rural']:
        subgroup_df = performance_df[performance_df['urban_rural_class'] == area_type]
        if not subgroup_df.empty:
            mae = mean_absolute_error(subgroup_df['actual'], subgroup_df['predicted'])
            rmse = np.sqrt(mean_squared_error(subgroup_df['actual'], subgroup_df['predicted']))
            results.append({
                'Model': model_prefix,
                'Area Type': area_type,
                'MAE': mae,
                'RMSE': rmse
            })
    return pd.DataFrame(results)

def calculate_degradation_v2(initial_val, degraded_val, is_higher_better=False):
    """
    Calculates percentage degradation between an initial and degraded value.

    Args:
        initial_val (float): Initial performance metric value.
        degraded_val (float): Degraded performance metric value.
        is_higher_better (bool): True if higher values are better (e.g., R-squared).

    Returns:
        float: Percentage degradation.
    """
    if initial_val is None or degraded_val is None or np.isnan(initial_val) or np.isnan(degraded_val):
        return np.nan

    if is_higher_better: # R-squared
        if initial_val == 1.0:
            return 0.0 if degraded_val == 1.0 else np.inf
        elif initial_val == 0.0:
            return 100.0 if degraded_val < 0.0 else 0.0
        elif initial_val < 0.0:
            return ((initial_val - degraded_val) / abs(initial_val)) * 100 if initial_val != 0 else np.nan
        else:
            return ((initial_val - degraded_val) / initial_val) * 100
    else: # MAE, RMSE, MAPE (lower is better)
        if initial_val == 0.0:
            return np.inf if degraded_val > 0 else 0.0
        return ((degraded_val - initial_val) / initial_val) * 100

def simulate_and_evaluate_sparsity(model_template, X_full, y_full, feature_cols_to_mask, masking_percentage, calculate_metrics_func):
    """
    Simulates data sparsity by masking features, imputing, retraining, and evaluating.

    Args:
        model_template (estimator): An initialized scikit-learn compatible model template.
        X_full (pd.DataFrame): Full feature set (unscaled).
        y_full (pd.Series): Full target variable.
        feature_cols_to_mask (list): List of feature column names to mask.
        masking_percentage (float): Percentage of values to mask (0.0 to 1.0).
        calculate_metrics_func (function): Function to calculate evaluation metrics.

    Returns:
        dict: Evaluation metrics after sparsity simulation.
    """
    X_sparse = X_full.copy()

    for col in feature_cols_to_mask:
        num_to_mask = int(len(X_sparse) * masking_percentage)
        mask_indices = np.random.choice(X_sparse.index, num_to_mask, replace=False)
        X_sparse.loc[mask_indices, col] = np.nan

    imputed_X_sparse = X_sparse.copy()
    for col in feature_cols_to_mask:
        median_val = imputed_X_sparse[col].median()
        imputed_X_sparse[col] = imputed_X_sparse[col].fillna(median_val)

    scaler = StandardScaler()
    scaler.fit(X_full)
    X_sparse_scaled = scaler.transform(imputed_X_sparse)
    X_sparse_scaled_df = pd.DataFrame(X_sparse_scaled, columns=X_full.columns, index=X_full.index)

    retrained_model = copy.deepcopy(model_template)
    retrained_model.fit(X_sparse_scaled_df, y_full)

    y_pred = retrained_model.predict(X_sparse_scaled_df)

    metrics = calculate_metrics_func(y_full, y_pred)
    return metrics

def plot_degradation_curve(degradation_df, model_name, metric_name, ax, y_label=None):
    """
    Plots a performance degradation curve.

    Args:
        degradation_df (pd.DataFrame): DataFrame containing degradation data.
        model_name (str): Name of the model.
        metric_name (str): Name of the metric to plot.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        y_label (str, optional): Label for the y-axis. Defaults to None.
    """
    ax.plot(degradation_df['Masking Percentage'], degradation_df[metric_name], marker='o', linestyle='-')
    ax.set_title(f'{model_name}: {metric_name} Degradation vs. Satellite Data Sparsity', fontsize=12)
    ax.set_xlabel('Percentage of Satellite Data Masked (%)')
    ax.set_ylabel(y_label if y_label else f'{metric_name} Degradation (%)')
    ax.grid(True)
    ax.set_ylim(bottom=min(degradation_df[metric_name].min(), 0) * 1.1)

def visualize_spatial_errors(block_groups_gdf, y_actual, y_pred, model_name, plot_title, output_filename=None):
    """
    Generates a choropleth map of prediction errors.

    Args:
        block_groups_gdf (gpd.GeoDataFrame): GeoDataFrame with block group geometries.
        y_actual (pd.Series): Actual income values.
        y_pred (array-like): Predicted income values.
        model_name (str): Name of the model.
        plot_title (str): Title for the map.
        output_filename (str, optional): Filename to save the map. Defaults to None.
    """
    residuals = y_actual - y_pred

    # Create a temporary DataFrame for merging residuals
    residuals_temp_df = pd.DataFrame({
        'GEOID': block_groups_gdf['GEOID'], # Assuming GEOID is in block_groups_gdf
        'prediction_error': residuals
    })

    # Ensure GEOID columns are consistent string types for merging
    block_groups_gdf['GEOID'] = block_groups_gdf['GEOID'].astype(str)
    residuals_temp_df['GEOID'] = residuals_temp_df['GEOID'].astype(str)

    error_map_gdf = block_groups_gdf.merge(residuals_temp_df, on='GEOID', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    error_map_gdf.plot(
        column='prediction_error',
        cmap='RdBu_r', # Red-Blue diverging colormap (red for underprediction, blue for overprediction)
        legend=True,
        legend_kwds={'label': f'Prediction Error ({model_name}) (Actual - Predicted) ($)', 'orientation': 'horizontal'},
        missing_kwds={'color': 'lightgrey', 'label': 'No Data'},
        ax=ax,
        vmax=abs(error_map_gdf['prediction_error']).max(), # Symmetrical color scale
        vmin=-abs(error_map_gdf['prediction_error']).max()  # Symmetrical color scale
    )
    ax.set_title(plot_title, fontsize=16)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    if output_filename: plt.savefig(output_filename, bbox_inches='tight')

def visualize_residuals_plot(y_pred, residuals, model_name, plot_title, output_filename=None):
    """
    Generates a residual plot (predicted vs. residuals).

    Args:
        y_pred (array-like): Predicted values.
        residuals (array-like): Residual values.
        model_name (str): Name of the model.
        plot_title (str): Title for the plot.
        output_filename (str, optional): Filename to save the plot. Defaults to None.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Predicted Median Income ($)')
    ax.set_ylabel('Residuals ($)')
    plt.tight_layout()
    plt.show()
    if output_filename: plt.savefig(output_filename, bbox_inches='tight')

def visualize_feature_importance(feature_importances_dict, model_name, top_n=10, output_filename=None):
    """
    Generates a bar chart of feature importances.

    Args:
        feature_importances_dict (dict): Dictionary of feature importances.
        model_name (str): Name of the model for the plot title.
        top_n (int): Number of top features to display.
        output_filename (str, optional): Filename to save the plot. Defaults to None.
    """
    feature_importances = pd.Series(feature_importances_dict).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances.head(top_n).plot(kind='barh', ax=ax, color='teal')
    ax.set_title(f'{model_name}: Top {top_n} Feature Importances', fontsize=16)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    if output_filename: plt.savefig(output_filename, bbox_inches='tight')
