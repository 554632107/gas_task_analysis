"""
混合气体分类
分类模型：随机森林
评价指标：准确率、F1值、混淆矩阵
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False   # 用于显示负号

# 设置路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'mixed_gas_classification_output')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')

# 创建输出目录
for directory in [OUTPUT_DIR, FIG_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"确保目录存在: {directory}")

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("===== 数据加载与预处理 =====")
    data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')
    print(f"加载数据文件: {data_file}")
    
    df = pd.read_csv(data_file, encoding='gbk')
    
    # 检查必要列是否存在
    required_columns = ['气体种类', '气体浓度(ppm)', '测量结果', '气体谱线(nm)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少必要列: '{col}'")
    
    # 创建组合标签
    df['气体种类_气体浓度'] = df['气体种类'] + '(' + df['气体浓度(ppm)'].astype(int).astype(str) + ')'
    print(f"类别数量: {df['气体种类_气体浓度'].nunique()}")
    
    # 标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['气体种类_气体浓度'])
    
    # 特征选择与标准化
    X = df[['测量结果', '气体谱线(nm)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")
    return X_train, X_test, y_train, y_test, label_encoder

def train_and_evaluate():
    """训练模型并评估性能"""
    print("\n===== 模型训练与评估 =====")
    
    # 加载数据
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()


    # 参数网格
    param_grid = {
    'n_estimators': [100, 150, 200],          # 决策树数量
    'max_depth': [10, 15, 20],          # 控制树的深度，防止过拟合
    'min_samples_split': [10, 20, 30, 40],    # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [5, 10, 20],          # 叶子节点最少样本数，防止过拟合
    'max_features': ['sqrt', 'log2'],         # 每次分裂考虑的特征数
    'bootstrap': [True],                      # 是否使用自助采样法
    'class_weight': ['balanced', None]        # 处理类别不平衡
    }


    # 网格搜索与交叉验证
    print("执行网格搜索...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    # 使用最佳模型评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    
    # 计算交叉验证结果
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"5折交叉验证准确率: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 保存关键结果
    save_results(accuracy, f1, cv_mean, cv_std, cm, label_encoder.classes_, grid_search.best_params_)
    
    # 可视化混淆矩阵
    plot_confusion_matrix(cm, label_encoder.classes_)
    
    # 保存模型
    save_model(best_model)
    
    print("\n===== 任务完成 =====")

def save_results(accuracy, f1, cv_mean, cv_std, cm, classes, best_params):
    """保存关键结果到CSV和JSON文件"""
    print("保存结果...")
    
    # 1. 保存性能指标到CSV
    results = pd.DataFrame({
        '指标': ['准确率', 'F1分数', '交叉验证准确率', '交叉验证标准差'],
        '值': [accuracy, f1, cv_mean, cv_std]
    })
    results.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), 
                  index=False, encoding='utf-8-sig')
    print(f"性能指标已保存到CSV文件: {os.path.join(OUTPUT_DIR, 'performance_metrics.csv')}")
    
    # 2. 保存混淆矩阵到CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'), 
                encoding='utf-8-sig')
    print(f"混淆矩阵已保存到CSV文件: {os.path.join(OUTPUT_DIR, 'confusion_matrix.csv')}")
    
    # 3. 保存完整结果到JSON
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
    
    print(f"完整结果已保存到JSON文件: {json_path}")

def plot_confusion_matrix(cm, classes):
    """混淆矩阵可视化-完整显示"""
    print(f"开始绘制混淆矩阵，类别数量: {len(classes)}")
    
    # 根据类别数量动态调整图表大小
    base_size = 0.8  # 每个类别的基础尺寸
    fig_width = max(10, base_size * len(classes))  # 最小宽度为10英寸
    fig_height = max(8, base_size * len(classes))  # 最小高度为8英寸
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # 绘制热力图
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                annot_kws={'size': max(6, 10 - len(classes)//10)},
                linewidths=.5)
    
    plt.title('混合气体分类混淆矩阵', fontsize=max(12, 14 - len(classes)//5))
    plt.xlabel('预测标签', fontsize=max(10, 12 - len(classes)//5))
    plt.ylabel('真实标签', fontsize=max(10, 12 - len(classes)//5))
    
    # 设置所有类别的标签
    plt.xticks(np.arange(len(classes)), classes, 
               rotation=45, 
               ha='right',
               fontsize=max(8, 10 - len(classes)//10))
    plt.yticks(np.arange(len(classes)), classes, 
               rotation=0,
               fontsize=max(8, 10 - len(classes)//10))
    
    # 确保标签完全可见
    plt.tight_layout()
    
    # 保存高分辨率图像
    plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵可视化已保存，尺寸: {fig_width:.1f}x{fig_height:.1f}英寸")

def save_model(model):
    """保存训练好的模型"""
    print("保存模型...")
    
    # 创建模型文件路径
    model_path = os.path.join(OUTPUT_DIR, 'gas_classification_model.joblib')
    
    # 保存模型
    dump(model, model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
