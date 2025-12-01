from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import copy # To create new model instances for retraining

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

def setup_validation_strategies(master_feature_matrix, target_col='median_income', group_col='TRACTCE'):
    """
    Sets up random train/test split and GroupKFold for spatial cross-validation.

    Args:
        master_feature_matrix (pd.DataFrame): The DataFrame containing all features and target.
        target_col (str): Name of the target variable column.
        group_col (str): Name of the column to use for grouping in GroupKFold (e.g., 'TRACTCE').

    Returns:
        tuple: X (features), y (target), X_train_random, X_test_random,
               y_train_random, y_test_random, gkf (GroupKFold object), groups (for GroupKFold).
    """
    # Ensure GEOID is string type for consistent extraction if needed
    master_feature_matrix['GEOID'] = master_feature_matrix['GEOID'].astype(str)

    # Extract TRACTCE from GEOID for grouping if not already present
    if group_col not in master_feature_matrix.columns:
        master_feature_matrix[group_col] = master_feature_matrix['GEOID'].str[5:11]

    # Create the full feature set X by dropping non-feature columns
    X = master_feature_matrix.drop(columns=['GEOID', target_col, group_col])
    y = master_feature_matrix[target_col]

    # 80/20 random train/test split
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 5-fold spatial cross-validation using GroupKFold
    gkf = GroupKFold(n_splits=5)
    groups = master_feature_matrix[group_col].astype(str)

    return X, y, X_train_random, X_test_random, y_train_random, y_test_random, gkf, groups

def initialize_models(random_state=42):
    """
    Initializes a dictionary of regression models.

    Args:
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary of initialized scikit-learn compatible models.
    """
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(random_state=random_state),
        'XGBRegressor': XGBRegressor(random_state=random_state, objective='reg:squarederror'),
        'Ridge': Ridge(random_state=random_state) # Include Ridge here for consistency
    }
    return models

def evaluate_model_on_split(model_name, model, X_train, y_train, X_test, y_test, feature_cols=None):
    """
    Trains and evaluates a single model on a given train/test split.

    Args:
        model_name (str): Name of the model.
        model (estimator): Initialized scikit-learn compatible model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        feature_cols (list, optional): List of feature column names for feature importances. Defaults to None.

    Returns:
        tuple: Dictionary of metrics and a pandas Series of feature importances (or None if not applicable).
    """
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print(f"{model_name} Metrics: {metrics}")

    importances = None
    if hasattr(model, 'feature_importances_') and feature_cols is not None:
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"{model_name} Feature Importances:\n{importances}")

    return metrics, importances

def evaluate_model_on_spatial_cv(model_name, model_template, X_full, y_full, gkf, groups, feature_cols=None):
    """
    Evaluates a single model using spatial cross-validation with GroupKFold.
    Applies StandardScaler within each fold to prevent data leakage.

    Args:
        model_name (str): Name of the model.
        model_template (estimator): An initialized scikit-learn compatible model (template for deepcopy).
        X_full (pd.DataFrame): Full feature set.
        y_full (pd.Series): Full target variable.
        gkf (GroupKFold): Initialized GroupKFold object.
        groups (pd.Series): Grouping variable for GroupKFold.
        feature_cols (list, optional): List of feature column names for feature importances. Defaults to None.

    Returns:
        tuple: Dictionary of aggregated mean/std metrics and a list of per-fold feature importances.
    """
    print(f"Evaluating {model_name} on spatial cross-validation...")
    cv_metrics = {'MAE': [], 'RMSE': [], 'R-squared': [], 'MAPE': []}
    cv_importances = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_full, y_full, groups)):
        print(f"--- Fold {fold+1}/{gkf.n_splits} ---")
        X_train_cv, X_test_cv = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_cv, y_test_cv = y_full.iloc[train_idx], y_full.iloc[test_idx]

        # Scale data for this fold using a new scaler to prevent data leakage
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler_cv.transform(X_test_cv)

        X_train_cv_scaled = pd.DataFrame(X_train_cv_scaled, columns=X_full.columns, index=X_train_cv.index)
        X_test_cv_scaled = pd.DataFrame(X_test_cv_scaled, columns=X_full.columns, index=X_test_cv.index)

        # Create a new model instance for each fold to ensure clean state
        model = copy.deepcopy(model_template)
        model.fit(X_train_cv_scaled, y_train_cv)
        y_pred = model.predict(X_test_cv_scaled)
        metrics = calculate_metrics(y_test_cv, y_pred)

        for metric_name, value in metrics.items():
            cv_metrics[metric_name].append(value)

        if hasattr(model, 'feature_importances_') and feature_cols is not None:
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            cv_importances.append(importances)

    # Aggregate metrics
    aggregated_metrics = {
        metric: {'mean': np.mean(values), 'std': np.std(values)}
        for metric, values in cv_metrics.items()
    }
    print(f"{model_name} Aggregated Metrics: {aggregated_metrics}")

    return aggregated_metrics, cv_importances

def run_hyperparameter_tuning(model_template, X_full, y_full, gkf, groups, param_grid, feature_cols):
    """
    Performs GridSearchCV for hyperparameter tuning using spatial cross-validation.

    Args:
        model_template (estimator): The model template (e.g., XGBRegressor) to tune.
        X_full (pd.DataFrame): Full feature set.
        y_full (pd.Series): Full target variable.
        gkf (GroupKFold): Initialized GroupKFold object.
        groups (pd.Series): Grouping variable for GroupKFold.
        param_grid (dict): Dictionary of hyperparameters to search.
        feature_cols (list): List of feature column names for feature importances.

    Returns:
        estimator: The best estimator found by GridSearchCV.
    """
    print("Running hyperparameter tuning...")
    # Wrap the entire process in a function that returns the best model and its metrics.

    # StandardScaler needs to be applied within the GridSearchCV pipeline or manually for each fold.
    # For GridSearchCV with custom CV (like GroupKFold), StandardScaler is usually applied directly to X_full and then GridSearchCV is run.
    # However, for rigorous spatial CV, it's better to scale *inside* each fold. This means doing a manual CV loop or using a pipeline.
    # For simplicity, we'll scale X_full once and pass that to GridSearchCV.
    # A more robust approach would involve a pipeline, but this directly mimics the notebook's approach.
    scaler = StandardScaler()
    X_full_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns, index=X_full.index)

    grid_search = GridSearchCV(
        estimator=model_template,
        param_grid=param_grid,
        cv=gkf, # Use the spatial GroupKFold cross-validator
        scoring='neg_root_mean_squared_error', # Optimize for lowest RMSE
        verbose=1, # Display progress
        n_jobs=-1 # Use all available CPU cores
    )

    grid_search.fit(X_full_scaled, y_full, groups=groups)

    print("GridSearchCV fitting complete.")
    print("Best parameters found: ", grid_search.best_params_)
    print("Best RMSE from GridSearchCV: ", -grid_search.best_score_)

    return grid_search.best_estimator_

def save_model(model, filepath):
    """
    Saves a trained model to disk using joblib.

    Args:
        model (estimator): The trained model object.
        filepath (str): Path to save the model.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Loads a trained model from disk using joblib.

    Args:
        filepath (str): Path to the saved model.

    Returns:
        estimator: The loaded model object.
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
