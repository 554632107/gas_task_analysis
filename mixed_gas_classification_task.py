
"""
Mixed gas classification
Model: Random Forest
Metrics: Accuracy, F1 score, Confusion Matrix
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, 
                            confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from joblib import dump

# Set Chinese font (if needed for display)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to display Chinese fonts
plt.rcParams['axes.unicode_minus'] = False   # Used to display minus sign

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'mixed_gas_classification_output')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Create output directories
for directory in [OUTPUT_DIR, FIG_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"Ensure directory exists: {directory}")

def load_and_preprocess_data():
    """Load and preprocess data"""
    print("===== Data loading and preprocessing =====")
    data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')
    print(f"Loading data file: {data_file}")
    
    df = pd.read_csv(data_file, encoding='gbk')
    
    # Validate required columns
    required_columns = ['气体种类', '气体浓度(ppm)', '测量结果', '气体谱线(nm)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Data file is missing required column: '{col}'")
    
    # Rename columns to English for downstream processing
    df = df.rename(columns={
        '气体种类': 'gas_type',
        '气体浓度(ppm)': 'gas_concentration_ppm',
        '测量结果': 'measurement_result',
        '气体谱线(nm)': 'gas_spectral_line_nm'
    })
    
    # Create combined label
    df['gas_type_concentration'] = df['gas_type'] + '(' + df['gas_concentration_ppm'].astype(int).astype(str) + ')'
    print(f"Number of classes: {df['gas_type_concentration'].nunique()}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['gas_type_concentration'])
    
    # Feature selection and standardization
    X = df[['measurement_result', 'gas_spectral_line_nm']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, label_encoder

def train_and_evaluate():
    """Train model and evaluate performance"""
    print("\n===== Model training and evaluation =====")
    
    # Load data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()


    # Parameter grid
    param_grid = {
        'n_estimators': [100, 150, 200],          # Number of trees
        'max_depth': [10, 15, 20],                # Control tree depth to prevent overfitting
        'min_samples_split': [10, 20, 30, 40],    # Minimum samples required to split an internal node
        'min_samples_leaf': [5, 10, 20],          # Minimum samples at a leaf node to prevent overfitting
        'max_features': ['sqrt', 'log2'],         # Number of features to consider at each split
        'bootstrap': [True],                      # Whether to use bootstrap sampling
        'class_weight': ['balanced', None]        # Handle class imbalance
    }


    # Grid search and cross-validation
    print("Running grid search...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    # Use best model for evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    
    # Calculate cross-validation results
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"5-fold CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save key results
    save_results(accuracy, f1, cv_mean, cv_std, cm, label_encoder.classes_, grid_search.best_params_)
    
    # Visualize confusion matrix
    plot_confusion_matrix(cm, label_encoder.classes_)
    
    # Save model
    save_model(best_model)
    
    print("\n===== Task completed =====")

def save_results(accuracy, f1, cv_mean, cv_std, cm, classes, best_params):
    """Save key results to CSV and JSON files"""
    print("Saving results...")
    
    # 1. Save performance metrics to CSV
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 score', 'CV accuracy', 'CV std'],
        'Value': [accuracy, f1, cv_mean, cv_std]
    })
    results.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), 
                  index=False, encoding='utf-8-sig')
    print(f"Performance metrics saved to CSV: {os.path.join(OUTPUT_DIR, 'performance_metrics.csv')}")
    
    # 2. Save confusion matrix to CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'), 
                encoding='utf-8-sig')
    print(f"Confusion matrix saved to CSV: {os.path.join(OUTPUT_DIR, 'confusion_matrix.csv')}")
    
    # 3. Save full results to JSON
    results_dict = {
        'best_parameters': best_params,
        'cross_validation_accuracy': {
            'mean': float(cv_mean),
            'std': float(cv_std)
        },
        'test_set_metrics': {
            'accuracy': float(accuracy),
            'f1_score': float(f1)
        }
    }
    
    json_path = os.path.join(OUTPUT_DIR, 'classification_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Full results saved to JSON: {json_path}")

def plot_confusion_matrix(cm, classes):
    """Confusion matrix visualization - full display"""
    print(f"Start plotting confusion matrix, number of classes: {len(classes)}")
    
    # Dynamically adjust figure size based on number of classes
    base_size = 0.8  # Base size per class
    fig_width = max(10, base_size * len(classes))  # Minimum width 10 inches
    fig_height = max(8, base_size * len(classes))  # Minimum height 8 inches
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Plot heatmap
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                annot_kws={'size': max(6, 10 - len(classes)//10)},
                linewidths=.5)
    
    plt.title('Confusion matrix for mixed gas classification', fontsize=max(12, 14 - len(classes)//5))
    plt.xlabel('Predicted label', fontsize=max(10, 12 - len(classes)//5))
    plt.ylabel('True label', fontsize=max(10, 12 - len(classes)//5))
    
    # Set all class labels
    plt.xticks(np.arange(len(classes)), classes, 
               rotation=45, 
               ha='right',
               fontsize=max(8, 10 - len(classes)//10))
    plt.yticks(np.arange(len(classes)), classes, 
               rotation=0,
               fontsize=max(8, 10 - len(classes)//10))
    
    # Ensure labels are fully visible
    plt.tight_layout()
    
    # Save high-resolution image
    plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix visualization saved, size: {fig_width:.1f}x{fig_height:.1f} inches")

def save_model(model):
    """Save trained model"""
    print("Saving model...")
    
    # Create model file path
    model_path = os.path.join(OUTPUT_DIR, 'gas_classification_model.joblib')
    
    # Save model
    dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
