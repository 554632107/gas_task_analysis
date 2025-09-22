"""
气体分类任务分析
功能：针对每种气体，分别进行二分类任务分析
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

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False   # 用于显示负号

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'gas_classification_results')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'detailed_results')

# 创建输出目录
for directory in [OUTPUT_DIR, FIG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"确保目录存在: {directory}")

def load_data(file_path):
    """加载数据，使用gbk编码（项目标准编码）"""
    print(f"开始加载数据文件: {file_path}")
    try:
        # 使用gbk编码
        df = pd.read_csv(file_path, encoding='gbk')
        print("成功使用gbk编码加载数据")
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}: {str(e)}")
    
    # 检查必要列是否存在
    required_columns = ['气体种类', '测量结果']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据文件缺少必要列: '{col}'")
    
    # 获取气体种类
    gas_types = df['气体种类'].unique()
    print(f"检测到 {len(gas_types)} 种气体分子: {', '.join(gas_types)}")
    
    return df, gas_types

def calculate_confidence_interval(cv_scores):
    """计算交叉验证结果的95%置信区间"""
    mean_accuracy = np.mean(cv_scores)
    # 使用t分布计算置信区间（小样本更准确）
    ci_lower = mean_accuracy - 2.776 * np.std(cv_scores) / np.sqrt(len(cv_scores))
    ci_upper = mean_accuracy + 2.776 * np.std(cv_scores) / np.sqrt(len(cv_scores))
    
    # 确保置信区间合理
    ci_lower = max(0, ci_lower)
    ci_upper = min(1, ci_upper)
    
    return ci_lower, ci_upper, mean_accuracy

def analyze_confidence(model, X_test, y_test):
    """分析模型预测的置信度指标"""
    # 检查模型是否支持predict_proba
    if not hasattr(model, 'predict_proba'):
        return {
            'avg_confidence': np.nan,
            'high_confidence_ratio': np.nan,
            'low_confidence_ratio': np.nan
        }
    
    # 获取预测概率
    confidences = np.max(model.predict_proba(X_test), axis=1)
    
    # 计算置信度指标
    avg_confidence = np.mean(confidences)
    high_confidence_ratio = np.mean(confidences >= 0.9)
    low_confidence_ratio = np.mean(confidences < 0.7)
    
    return {
        'avg_confidence': avg_confidence,
        'high_confidence_ratio': high_confidence_ratio,
        'low_confidence_ratio': low_confidence_ratio
    }

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """评估分类模型并返回关键指标，包含必要的置信度数值"""
    # 在训练集上进行5折交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    ci_lower, ci_upper, cv_mean = calculate_confidence_interval(cv_scores)
    
    # 在测试集上评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 二分类任务指标计算
    f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # 置信度分析
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
    """针对单种气体进行二分类分析"""
    print(f"\n开始分析气体: {gas_type}")
    
    # 筛选该气体数据
    gas_mask = (df['气体种类'] == gas_type)
    num_samples = gas_mask.sum()
    print(f"气体 {gas_type} 的样本数量: {num_samples}")
    
    # 检查是否有足够样本
    if num_samples < 10:
        print(f"警告: 气体 {gas_type} 的样本数量过少 ({num_samples}), 跳过分析")
        return None
    
    # 创建分类标签
    y_binary = gas_mask.astype(int).values
    
    # 检查是否包含两类样本
    if len(np.unique(y_binary)) < 2:
        print(f"警告: 气体 {gas_type} 的二分类任务中只有一种类别，跳过此任务")
        return None
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    train_size, test_size = len(X_train), len(X_test)
    print(f"训练集样本: {train_size}, 测试集样本: {test_size}")
    
    # 训练逻辑回归模型
    print(f"  训练逻辑回归模型...")
    lr_model = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
    
    # 训练随机森林模型
    print(f"  训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
    
    # 保存混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, rf_model.predict(X_test))
    # 使用matplotlib绘制热力图替代seaborn
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{gas_type} 混淆矩阵')
    plt.colorbar()
    # 添加数值标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(FIG_DIR, f'{gas_type}_confusion.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 格式化结果，保留4位小数
    results = []
    for model_name, model_results in [('逻辑回归', lr_results), ('随机森林', rf_results)]:
        formatted = {}
        for k, v in model_results.items():
            if not pd.isna(v):
                # 保留4位小数
                formatted[k] = round(float(v), 4) if isinstance(v, (int, float)) else v
            else:
                formatted[k] = v
        formatted.update({'气体类型': gas_type, '模型': model_name})
        results.append(formatted)
    
    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, f'{gas_type}_results.csv'), 
                     index=False, encoding='utf-8-sig')
    
    # 打印关键置信度指标
    print(f"  {gas_type} 置信度指标:")
    print(f"    逻辑回归 - 平均置信度: {lr_results['avg_confidence']:.4f}, "
          f"高置信度比例: {lr_results['high_confidence_ratio']:.4f}, "
          f"低置信度比例: {lr_results['low_confidence_ratio']:.4f}")
    print(f"    随机森林 - 平均置信度: {rf_results['avg_confidence']:.4f}, "
          f"高置信度比例: {rf_results['high_confidence_ratio']:.4f}, "
          f"低置信度比例: {rf_results['low_confidence_ratio']:.4f}")
    
    print(f"  {gas_type}分类分析完成，结果已保存")
    return results_df

def main():
    # 设置数据文件路径
    data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')
    print(f"\n===== 气体分类任务分析开始 =====")
    print(f"数据文件路径: {data_file}")
    
    # 加载数据
    df, gas_types = load_data(data_file)
    
    # 特征预处理 
    print("\n开始特征预处理...")
    X = df[['测量结果']]
    # X = df[['测量结果', '气体谱线(nm)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("特征标准化完成")
    
    # 为每种气体单独分析
    all_results = []
    for gas_type in gas_types:
        results = analyze_gas_type(df, X_scaled, gas_type)
        if results is not None:
            all_results.append(results)
    
    if not all_results:
        print("\n警告: 没有生成任何有效结果")
        return
    
    # 生成汇总报告
    summary = pd.concat(all_results, ignore_index=True)
    
    # 保存汇总结果
    summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_results.csv'), 
                  index=False, encoding='utf-8-sig')
    print(f"\n汇总结果已保存至: {os.path.join(OUTPUT_DIR, 'summary_results.csv')}")
    
    # 可视化整体结果（仅保留核心指标）
    plt.figure(figsize=(12, 8))
    
    # 准确率比较（带置信区间）
    plt.subplot(2, 2, 1)
    for gas_type in gas_types:
        gas_data = summary[summary['气体类型'] == gas_type]
        for model in ['逻辑回归', '随机森林']:
            model_data = gas_data[gas_data['模型'] == model]
            yerr_lower = model_data['accuracy'] - model_data['ci_lower']
            yerr_upper = model_data['ci_upper'] - model_data['accuracy']
            plt.errorbar(
                gas_type + f'_{model}', 
                model_data['accuracy'],
                yerr=[np.maximum(0, yerr_lower), np.maximum(0, yerr_upper)],
                fmt='o-', capsize=5
            )
    plt.title('准确率比较（带95%置信区间）')
    plt.ylabel('准确率')
    plt.xticks(rotation=45)
    
    # F1分数比较
    plt.subplot(2, 2, 2)
    for gas_type in gas_types:
        gas_data = summary[summary['气体类型'] == gas_type]
        for model in ['逻辑回归', '随机森林']:
            model_data = gas_data[gas_data['模型'] == model]
            plt.bar(f'{gas_type}_{model}', model_data['f1_score'], alpha=0.7)
    plt.title('F1分数比较')
    plt.ylabel('F1分数')
    plt.xticks(rotation=45)
    
    # 平均置信度比较
    plt.subplot(2, 2, 3)
    for gas_type in gas_types:
        gas_data = summary[summary['气体类型'] == gas_type]
        for model in ['逻辑回归', '随机森林']:
            model_data = gas_data[gas_data['模型'] == model]
            plt.bar(f'{gas_type}_{model}', model_data['avg_confidence'], alpha=0.7)
    plt.title('平均置信度比较')
    plt.ylabel('置信度')
    plt.xticks(rotation=45)
    
    # 低置信度比例
    plt.subplot(2, 2, 4)
    for gas_type in gas_types:
        gas_data = summary[summary['气体类型'] == gas_type]
        for model in ['逻辑回归', '随机森林']:
            model_data = gas_data[gas_data['模型'] == model]
            plt.bar(f'{gas_type}_{model}', model_data['low_confidence_ratio'], alpha=0.7)
    plt.title('低置信度样本比例（<0.7）')
    plt.ylabel('比例')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overall_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存至: {FIG_DIR}")
    print("\n===== 气体分类任务分析完成 =====")

if __name__ == "__main__":
    main()
