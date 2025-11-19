"""
Gas classification task analysis
Function: Perform binary classification analysis for each gas type separately
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to display Chinese fonts if needed
plt.rcParams['axes.unicode_minus'] = False   # Used to display minus sign

# Configuration paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'gas_classification_results')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'detailed_results')

# Create output directories
for directory in [OUTPUT_DIR, FIG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"Ensure directory exists: {directory}")

def load_data(file_path):
    """Load data using gbk encoding (project standard encoding)"""
    print(f"Start loading data file: {file_path}")
    try:
        # Use gbk encoding
        df = pd.read_csv(file_path, encoding='gbk')
        print("Successfully loaded data using gbk encoding")
    except Exception as e:
        raise ValueError(f"Unable to read file {file_path}: {str(e)}")
    
    # Check if required columns exist
    required_columns = ['gas_type', 'measurement_result']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Data file is missing required column: '{col}'")
    
    # Get gas types
    gas_types = df['gas_type'].unique()
    print(f"Detected {len(gas_types)} gas molecules: {', '.join(gas_types)}")
    
    return df, gas_types

def calculate_confidence_interval(cv_scores):
    """Calculate 95% confidence interval of cross-validation results"""
    mean_accuracy = np.mean(cv_scores)
    # Use t-distribution to compute CI (more accurate for small samples)
    ci_lower = mean_accuracy - 2.776 * np.std(cv_scores) / np.sqrt(len(cv_scores))
    ci_upper = mean_accuracy + 2.776 * np.std(cv_scores) / np.sqrt(len(cv_scores))
    
    # Ensure CI is reasonable
    ci_lower = max(0, ci_lower)
    ci_upper = min(1, ci_upper)
    
    return ci_lower, ci_upper, mean_accuracy

def analyze_confidence(model, X_test, y_test):
    """Analyze confidence metrics of model predictions"""
    # Check if model supports predict_proba
    if not hasattr(model, 'predict_proba'):
        return {
            'avg_confidence': np.nan,
            'high_confidence_ratio': np.nan,
            'low_confidence_ratio': np.nan
        }
    
    # Get prediction probabilities
    confidences = np.max(model.predict_proba(X_test), axis=1)
    
    # Compute confidence metrics
    avg_confidence = np.mean(confidences)
    high_confidence_ratio = np.mean(confidences >= 0.9)
    low_confidence_ratio = np.mean(confidences < 0.7)
    
    return {
        'avg_confidence': avg_confidence,
        'high_confidence_ratio': high_confidence_ratio,
        'low_confidence_ratio': low_confidence_ratio
    }

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate classification model and return key metrics, including necessary confidence values"""
    # Perform 5-fold cross-validation on the training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    ci_lower, ci_upper, cv_mean = calculate_confidence_interval(cv_scores)
    
    # Evaluate on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute binary classification metrics
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Confidence analysis
    confidence_metrics = analyze_confidence(model, X_test, y_test)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'cv_mean': cv_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        **confidence_metrics
    }

def analyze_gas_type(df, X_scaled, gas_type):
    """Perform binary classification analysis for a single gas type"""
    print(f"\nStart analyzing gas: {gas_type}")
    
    # Filter data for this gas
    gas_mask = (df['gas_type'] == gas_type)
    num_samples = gas_mask.sum()
    print(f"Gas {gas_type} sample count: {num_samples}")
    
    # Check if there are enough samples
    if num_samples < 10:
        print(f"Warning: Gas {gas_type} has too few samples ({num_samples}), skipping analysis")
        return None
    
    # Create binary labels
    y_binary = gas_mask.astype(int).values
    
    # Check if both classes are present
    if len(np.unique(y_binary)) < 2:
        print(f"Warning: Binary task for gas {gas_type} has only one class, skipping this task")
        return None
    
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    train_size, test_size = len(X_train), len(X_test)
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    # Train Logistic Regression model
    print(f"  Training Logistic Regression model...")
    lr_model = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
    
    # Train Random Forest model
    print(f"  Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
    
    # Save confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, rf_model.predict(X_test))
    # Use matplotlib heatmap instead of seaborn
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{gas_type} Confusion Matrix')
    plt.colorbar()
    # Add value annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(FIG_DIR, f'{gas_type}_confusion.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Format results, keep 4 decimal places
    results = []
    for model_name, model_results in [('Logistic Regression', lr_results), ('Random Forest', rf_results)]:
        formatted = {}
        for k, v in model_results.items():
            if not pd.isna(v):
                # Keep 4 decimal places
                formatted[k] = round(float(v), 4) if isinstance(v, (int, float)) else v
            else:
                formatted[k] = v
        formatted.update({'gas_type': gas_type, 'model': model_name})
        results.append(formatted)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, f'{gas_type}_results.csv'), 
                      index=False, encoding='utf-8-sig')
    
    # Print key confidence metrics
    print(f"  {gas_type} Confidence metrics:")
    print(f"    Logistic Regression - Average confidence: {lr_results['avg_confidence']:.4f}, "
          f"High-confidence ratio: {lr_results['high_confidence_ratio']:.4f}, "
          f"Low-confidence ratio: {lr_results['low_confidence_ratio']:.4f}")
    print(f"    Random Forest - Average confidence: {rf_results['avg_confidence']:.4f}, "
          f"High-confidence ratio: {rf_results['high_confidence_ratio']:.4f}, "
          f"Low-confidence ratio: {rf_results['low_confidence_ratio']:.4f}")
    
    print(f"  {gas_type} classification analysis completed, results saved")
    return results_df

def main():
    # Set data file path
    data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')
    print(f"\n===== Gas classification analysis begins =====")
    print(f"Data file path: {data_file}")
    
    # Load data
    df, gas_types = load_data(data_file)
    
    # Feature preprocessing 
    print("\nStart feature preprocessing...")
    X = df[['measurement_result']]
    # X = df[['measurement_result', 'gas spectral line (nm)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature standardization completed")
    
    # Analyze each gas separately
    all_results = []
    for gas_type in gas_types:
        results = analyze_gas_type(df, X_scaled, gas_type)
        if results is not None:
            all_results.append(results)
    
    if not all_results:
        print("\nWarning: No valid results generated")
        return
    
    # Generate summary report
    summary = pd.concat(all_results, ignore_index=True)
    
    # Save summary results
    summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_results.csv'), 
                   index=False, encoding='utf-8-sig')
    print(f"\nSummary results saved to: {os.path.join(OUTPUT_DIR, 'summary_results.csv')}")
    
    # Visualize overall results (keep core metrics only)
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison (with confidence interval)
    plt.subplot(2, 2, 1)
    for gas_type in gas_types:
        gas_data = summary[summary['gas_type'] == gas_type]
        for model in ['Logistic Regression', 'Random Forest']:
            model_data = gas_data[gas_data['model'] == model]
            yerr_lower = model_data['accuracy'] - model_data['ci_lower']
            yerr_upper = model_data['ci_upper'] - model_data['accuracy']
            plt.errorbar(
                gas_type + f'_{model}', 
                model_data['accuracy'],
                yerr=[np.maximum(0, yerr_lower), np.maximum(0, yerr_upper)],
                fmt='o-', capsize=5
            )
    plt.title('Accuracy comparison (with 95% CI)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # F1 score comparison
    plt.subplot(2, 2, 2)
    for gas_type in gas_types:
        gas_data = summary[summary['gas_type'] == gas_type]
        for model in ['Logistic Regression', 'Random Forest']:
            model_data = gas_data[gas_data['model'] == model]
            plt.bar(f'{gas_type}_{model}', model_data['f1_score'], alpha=0.7)
    plt.title('F1 score comparison')
    plt.ylabel('F1 score')
    plt.xticks(rotation=45)
    
    # Average confidence comparison
    plt.subplot(2, 2, 3)
    for gas_type in gas_types:
        gas_data = summary[summary['gas_type'] == gas_type]
        for model in ['Logistic Regression', 'Random Forest']:
            model_data = gas_data[gas_data['model'] == model]
            plt.bar(f'{gas_type}_{model}', model_data['avg_confidence'], alpha=0.7)
    plt.title('Average confidence comparison')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    
    # Low-confidence ratio
    plt.subplot(2, 2, 4)
    for gas_type in gas_types:
        gas_data = summary[summary['gas_type'] == gas_type]
        for model in ['Logistic Regression', 'Random Forest']:
            model_data = gas_data[gas_data['model'] == model]
            plt.bar(f'{gas_type}_{model}', model_data['low_confidence_ratio'], alpha=0.7)
    plt.title('Low-confidence sample ratio (<0.7)')
    plt.ylabel('Ratio')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overall_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization results saved to: {FIG_DIR}")
    print("\n===== Gas classification analysis completed =====")

if __name__ == "__main__":
    main()

