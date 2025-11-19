"""
Gas concentration regression analysis script
Core task: For each gas, perform concentration regression prediction analysis, confidence interval calculation, and model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from scipy import stats

# Set font for plots
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Configure paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'gas_regression_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    """Load data, only try common encodings"""
    print("=> Loading data...")
    try:
        # Prefer utf-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
        print("Loaded data using utf-8 encoding")
    except:
        try:
            # Fallback to gbk encoding
            df = pd.read_csv(file_path, encoding='gbk')
            print("Loaded data using gbk encoding")
        except Exception as e:
            raise ValueError(f"Unable to read file {file_path}: {str(e)}")
    
    # Validate required columns (in raw Chinese column names)
    required_columns = ['气体种类', '气体浓度(ppm)', '测量结果', '气体谱线(nm)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data file is missing required columns: {', '.join(missing_cols)}")
    
    # Rename columns to English for downstream processing
    df = df.rename(columns={
        '气体种类': 'gas_type',
        '气体浓度(ppm)': 'gas_concentration_ppm',
        '测量结果': 'measurement_result',
        '气体谱线(nm)': 'gas_spectral_line_nm'
    })
    
    # Extract unique gas types and concentration values
    gas_types = df['gas_type'].unique()
    concentrations = sorted(df['gas_concentration_ppm'].unique())
    
    print(f"Detected {len(gas_types)} gas types: {', '.join(gas_types)}")
    print(f"Detected {len(concentrations)} concentration levels: {', '.join(map(str, concentrations))}")
    
    return df, gas_types, concentrations

def calculate_confidence_interval(scores, confidence=0.95):
    """Calculate confidence interval"""
    if len(scores) < 2:
        return np.nan, np.nan
    
    mean_score = np.mean(scores)
    std_err = stats.sem(scores)
    
    # Calculate t-distribution critical value
    t_critical = stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    margin_of_error = t_critical * std_err
    
    ci_lower = max(0, mean_score - margin_of_error)
    ci_upper = mean_score + margin_of_error
    
    return ci_lower, ci_upper



def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, gas_type):
    """Evaluate regression model and return key metrics"""
    print(f"   => Evaluating {model_name} model...")
    
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate basic metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation to calculate MAE confidence interval
    try:
        cv_results = cross_validate(model, X_train, y_train, cv=5, 
                                  scoring='neg_mean_absolute_error')
        mae_scores = -cv_results['test_score']
        mae_ci_lower, mae_ci_upper = calculate_confidence_interval(mae_scores)
    except Exception as e:
        print(f"   ! Cross-validation error: {str(e)}")
        mae_ci_lower, mae_ci_upper = np.nan, np.nan
    
    # Calculate 95% prediction interval coverage
    errors = y_test - y_pred
    ci_95 = 1.96 * np.std(errors)
    coverage = np.mean((y_test >= (y_pred - ci_95)) & (y_test <= (y_pred + ci_95)))
    
    # Print key results
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE 95% CI: [{mae_ci_lower:.4f}, {mae_ci_upper:.4f}]")
    print(f"95% prediction interval coverage: {coverage:.4f}")
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAE_CI_lower': mae_ci_lower,
        'MAE_CI_upper': mae_ci_upper,
        '95%_interval': ci_95,
        'coverage': coverage
    }

def analyze_gas_type(df, X, y, gas_type):
    """Perform concentration regression analysis for a single gas type"""
    print(f"\n=> Start analyzing gas: {gas_type}")
    
    # Extract data for this gas
    gas_mask = (df['gas_type'] == gas_type)
    X_gas = X[gas_mask]
    y_gas = y[gas_mask]
    
    # Check data volume
    if len(X_gas) < 10:
        print(f"   ! Warning: {gas_type} insufficient data ({len(X_gas)} records), skipping analysis")
        return None
    
    print(f"Retrieved {gas_type} data: {len(X_gas)} records")
    

    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_gas, y_gas, test_size=0.2, random_state=42
    )
    
    # Initialize models
    lr_reg = LinearRegression()
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Evaluate models
    lr_results = evaluate_model(lr_reg, X_train, X_test, y_train, y_test, "Linear Regression", gas_type)
    rf_results = evaluate_model(rf_reg, X_train, X_test, y_train, y_test, "Random Forest", gas_type)
    
    # Save results
    results = pd.DataFrame([
        {'Model': 'Linear Regression', **lr_results},
        {'Model': 'Random Forest', **rf_results}
    ])
    
    # Generate key prediction figure
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_reg.predict(X_test), alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('True concentration (ppm)')
    plt.ylabel('Predicted concentration (ppm)')
    plt.title(f'{gas_type} Random Forest prediction performance')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{gas_type}_prediction.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {gas_type} prediction chart")
    
    return results

def main():
    print("===== Gas concentration regression analysis begins =====")
    
    # 1. Load data
    data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')
    try:
        df, gas_types, concentrations = load_data(data_file)
        print("\n=> Data loaded")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return
    
    # 2. Data preprocessing
    print("\n=> Start data preprocessing...")
    X = df[['measurement_result', 'gas_spectral_line_nm']]
    y = df['gas_concentration_ppm'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature standardization completed")
    
    # 3. Analyze by gas type
    all_results = []
    for gas_type in gas_types:
        results = analyze_gas_type(df, X_scaled, y, gas_type)
        if results is not None:
            all_results.append((gas_type, results))
    
    if not all_results:
        print("\n! All gas analyses failed, task terminated")
        return
    
    # 4. Summary results
    print("\n=> Generating summary results...")
    summary_data = []
    for gas_type, results in all_results:
        for _, row in results.iterrows():
            row_dict = row.to_dict()
            row_dict['Gas Type'] = gas_type
            summary_data.append(row_dict)
    
    summary = pd.DataFrame(summary_data)
    
    # Save summary results
    summary.to_csv(os.path.join(OUTPUT_DIR, 'regression_summary.csv'), 
                  index=False, encoding='utf-8-sig')
    print("Summary saved to regression_summary.csv")
    
    # 5. Generate key charts
    print("\n=> Generating key analysis charts...")
    
    # MAE comparison chart
    plt.figure(figsize=(10, 6))
    for gas_type in gas_types:
        gas_data = summary[summary['Gas Type'] == gas_type]
        for model in ['Linear Regression', 'Random Forest']:
            model_data = gas_data[gas_data['Model'] == model]
            mae = model_data['MAE'].values[0]
            plt.bar(f'{model}\n{gas_type}', mae, alpha=0.7)
    
    plt.title('MAE comparison')
    plt.ylabel('MAE (ppm)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Key charts generated")
    
    # 6. Print final summary
    print("\n===== Analysis results summary =====")
    print(f"Successfully analyzed {len(all_results)} gas types")
    
    # Show best model by gas type
    for gas_type in gas_types:
        gas_data = summary[summary['Gas Type'] == gas_type]
        if not gas_data.empty:
            best_model = gas_data.loc[gas_data['MAE'].idxmin(), 'Model']
            best_mae = gas_data['MAE'].min()
            print(f"- {gas_type}: Best model is {best_model} (MAE={best_mae:.4f})")
    
    print("\n===== Gas concentration regression analysis completed =====")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
