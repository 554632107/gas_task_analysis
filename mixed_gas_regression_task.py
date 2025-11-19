
"""
Mixed gas concentration regression analysis
Regression model: Random Forest Regressor
Metrics: RMSE, MAE, R2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from joblib import dump
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # Add more font options
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Set project root directory and output directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'mixed_gas_regression_output')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Data loading and preprocessing
data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')  # Use relative path
print(f"Loading data file: {data_file}")
df = pd.read_csv(data_file, encoding='gbk')
# Rename columns to English for downstream processing
df = df.rename(columns={
    '测量结果': 'measurement_result',
    '气体谱线(nm)': 'gas_spectral_line_nm',
    '气体浓度(ppm)': 'gas_concentration_ppm'
})

# Check for missing values
print("Missing values stats:\n", df.isnull().sum())

# 2. Dataset splitting
# Regression task (predict gas concentration)
X_reg = df[['measurement_result', 'gas_spectral_line_nm']]  # use original features
y_reg = df['gas_concentration_ppm']

# Split into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# Standardization of numerical variables
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# 3. Regression model optimization - Grid search and cross-validation
print("\n=== Regression model optimization ===")

# Define grid parameters
param_grid = {
    'n_estimators': [100, 150, 200, 250],          # Number of trees
    'max_depth': [15, 20, 25],                     # Max tree depth
    'min_samples_split': [20, 30, 50],             # Min samples to split a node
    'min_samples_leaf': [10, 15, 20],              # Min samples at leaf
    'max_features': ['auto', 'sqrt']               # Features considered at split
}


# Create Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf_regressor,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Train grid search model
print("Starting grid search...")
grid_search.fit(X_train_reg_scaled, y_train_reg)

# Output best parameters
print("\nBest parameters:", grid_search.best_params_)
print("Best CV MSE: {:.4f}".format(-grid_search.best_score_))

# Use best model for prediction
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_reg_scaled)

# 4. Model evaluation
print("\n=== Regression model evaluation ===")

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
mae = mean_absolute_error(y_test_reg, y_pred_rf)
r2 = r2_score(y_test_reg, y_pred_rf)

print("Test RMSE: {:.4f}".format(rmse))
print("Test MAE: {:.4f}".format(mae))
print("Test R2: {:.4f}".format(r2))

# 5. 95% prediction interval
errors = y_test_reg - y_pred_rf
ci_95 = 1.96 * np.std(errors)
coverage = np.mean((y_test_reg >= (y_pred_rf - ci_95)) & 
                  (y_test_reg <= (y_pred_rf + ci_95)))
print(f"95% prediction interval width: {ci_95:.4f}")
print(f"Coverage: {coverage:.4f}")

# 6. Error analysis grouped by concentration
concentrations = sorted(y_test_reg.unique())
print("\n=== Error analysis grouped by concentration ===")
print("\nMAE grouped by concentration:")
for conc in concentrations:
    mask = (y_test_reg == conc)
    if np.sum(mask) > 0:
        mae = mean_absolute_error(y_test_reg[mask], y_pred_rf[mask])
        print(f"MAE at {conc} ppm: {mae:.4f}")

# 7. Feature importance analysis
print("\n=== Feature importance analysis ===")
feature_importance = pd.DataFrame({
    'Feature': X_reg.columns,
    'Importance': best_rf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature importance for concentration regression model', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'concentration_feature_importance.png'), dpi=300, bbox_inches='tight')
# plt.show()

# 8. 5-fold cross-validation analysis
print("\n=== 5-fold cross-validation analysis ===")
from sklearn.model_selection import cross_validate, KFold

def plot_cross_validation_results(estimator, X, y, cv=None, n_jobs=-1):
    plt.figure(figsize=(12, 10))
    
    # Execute 5-fold cross-validation, get multiple metrics
    cv_results = cross_validate(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=True
    )
    
    # Get results from each fold
    train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'])
    test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])
    train_mae = -cv_results['train_neg_mean_absolute_error']
    test_mae = -cv_results['test_neg_mean_absolute_error']
    train_r2 = cv_results['train_r2']
    test_r2 = cv_results['test_r2']
    
    # Create fold numbers
    fold_numbers = np.arange(1, len(train_rmse) + 1)
    
    # Plot RMSE results
    plt.subplot(3, 1, 1)
    plt.plot(fold_numbers, train_rmse, 'o-', color='r', label='Train RMSE')
    plt.plot(fold_numbers, test_rmse, 'o-', color='g', label='Validation RMSE')
    
    # Add numerical annotations (keep 4 decimal places)
    for i, (train_r, test_r) in enumerate(zip(train_rmse, test_rmse)):
        plt.annotate(f'{train_r:.4f}', (fold_numbers[i], train_r), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{test_r:.4f}', (fold_numbers[i], test_r), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.xlabel('Fold number', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('5-fold CV - RMSE', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fold_numbers, fold_numbers)
    
    # Plot MAE results
    plt.subplot(3, 1, 2)
    plt.plot(fold_numbers, train_mae, 'o-', color='r', label='Train MAE')
    plt.plot(fold_numbers, test_mae, 'o-', color='g', label='Validation MAE')
    
    # Add numerical annotations (keep 4 decimal places)
    for i, (train_m, test_m) in enumerate(zip(train_mae, test_mae)):
        plt.annotate(f'{train_m:.4f}', (fold_numbers[i], train_m), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{test_m:.4f}', (fold_numbers[i], test_m), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.xlabel('Fold number', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('5-fold CV - MAE', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fold_numbers, fold_numbers)
    
    # Plot R2 results
    plt.subplot(3, 1, 3)
    plt.plot(fold_numbers, train_r2, 'o-', color='r', label='Train R2')
    plt.plot(fold_numbers, test_r2, 'o-', color='g', label='Validation R2')
    
    # Add numerical annotations (keep 4 decimal places)
    for i, (train_r2_val, test_r2_val) in enumerate(zip(train_r2, test_r2)):
        plt.annotate(f'{train_r2_val:.4f}', (fold_numbers[i], train_r2_val), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{test_r2_val:.4f}', (fold_numbers[i], test_r2_val), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.xlabel('Fold number', fontsize=12)
    plt.ylabel('R2', fontsize=12)
    plt.title('5-fold CV - R2', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fold_numbers, fold_numbers)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print cross-validation results
    print("\n5-fold CV results:")
    print("Fold numbers:", fold_numbers)
    print("Train RMSE:", [f'{rmse:.4f}' for rmse in train_rmse])
    print("Validation RMSE:", [f'{rmse:.4f}' for rmse in test_rmse])
    print("Train MAE:", [f'{mae:.4f}' for mae in train_mae])
    print("Validation MAE:", [f'{mae:.4f}' for mae in test_mae])
    print("Train R2:", [f'{r2:.4f}' for r2 in train_r2])
    print("Validation R2:", [f'{r2:.4f}' for r2 in test_r2])
    
    # Calculate and print statistics
    print("\nStatistics:")
    print(f"Validation RMSE - mean: {np.mean(test_rmse):.4f}, std: {np.std(test_rmse):.4f}")
    print(f"Validation MAE - mean: {np.mean(test_mae):.4f}, std: {np.std(test_mae):.4f}")
    print(f"Validation R2 - mean: {np.mean(test_r2):.4f}, std: {np.std(test_r2):.4f}")
    
    # Save cross-validation data to file
    cv_data = pd.DataFrame({
        'Fold': fold_numbers,
        'Train_RMSE': train_rmse,
        'Validation_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Validation_MAE': test_mae,
        'Train_R2': train_r2,
        'Validation_R2': test_r2
    })
    
    # Save as CSV file
    csv_path = os.path.join(OUTPUT_DIR, 'cross_validation_data.csv')
    cv_data.to_csv(csv_path, index=False, encoding='gbk')
    print(f"\nCross-validation data saved to CSV file: {csv_path}")
    
    # Save as Excel file
    excel_path = os.path.join(OUTPUT_DIR, 'cross_validation_data.xlsx')
    cv_data.to_excel(excel_path, index=False)
    print(f"Cross-validation data saved to Excel file: {excel_path}")
    
    return cv_results

# Run 5-fold cross-validation
print("Running 5-fold cross-validation...")
cv_results = plot_cross_validation_results(
    best_rf, X_train_reg_scaled, y_train_reg, 
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)



# Predicted vs True scatter plot
print("\n=== Predicted vs True visualization ===")

plt.figure(figsize=(10, 8))
plt.scatter(y_test_reg, y_pred_rf, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('True concentration (ppm)', fontsize=12)
plt.ylabel('Predicted concentration (ppm)', fontsize=12)
plt.title('Predicted vs True for concentration regression model', fontsize=14, fontweight='bold')

# Modify text display to avoid superscript characters
r2_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.4f}'
plt.text(0.05, 0.95, r2_text,
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'concentration_prediction_scatter.png'), dpi=300, bbox_inches='tight')
# plt.show()

# Model saving
model_path = os.path.join(OUTPUT_DIR, 'best_rf_model.joblib')
dump(best_rf, model_path)
print(f"\nBest model saved to: {model_path}")

# Save results to file
results = {
    'best_params': grid_search.best_params_,
    'best_cv_mse': -grid_search.best_score_,
    'test_rmse': rmse,
    'test_mae': mae,
    'test_r2': r2,
    'ci_95': ci_95,
    'coverage': coverage,
    'feature_importances': dict(zip(X_reg.columns, best_rf.feature_importances_))
}

# Save as JSON file
import json
results_json_path = os.path.join(OUTPUT_DIR, 'model_results.json')
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"\nModel results saved to JSON file: {results_json_path}")

print("\nConcentration regression analysis completed!")
