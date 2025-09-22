"""
混合气体浓度回归分析
回归模型：随机森林回归
评价指标：RMSE、MAE、R2
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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 添加更多字体选项
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 设置项目根目录和输出目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'mixed_gas_regression_output')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 1. 数据加载与预处理
data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')  # 使用相对路径
print(f"正在加载数据文件: {data_file}")
df = pd.read_csv(data_file, encoding='gbk')

# 检查缺失值
print("缺失值统计:\n", df.isnull().sum())

# 2. 数据集划分
# 回归任务（预测气体浓度）
X_reg = df[['测量结果', '气体谱线(nm)']]  # 仅使用原始特征
y_reg = df['气体浓度(ppm)']

# 划分训练集和测试集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# 数值变量的标准化处理
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# 3. 回归模型优化 - 网格搜索与交叉验证
print("\n=== 回归模型优化 ===")

# 定义网格参数
param_grid = {
    'n_estimators': [100, 150, 200, 250],          # 决策树数量
    'max_depth': [15, 20, 25],          # 树的最大深度
    'min_samples_split': [20, 30, 50],        # 分裂所需最小样本数
    'min_samples_leaf': [10, 15, 20],         # 叶子节点最小样本数
    'max_features': ['auto', 'sqrt']         # 每次分裂考虑的特征数
}


# 创建随机森林回归器
rf_regressor = RandomForestRegressor(random_state=42)

# 网格搜索与5折交叉验证
grid_search = GridSearchCV(
    estimator=rf_regressor,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# 训练网格搜索模型
print("开始网格搜索...")
grid_search.fit(X_train_reg_scaled, y_train_reg)

# 输出最佳参数
print("\n最佳参数:", grid_search.best_params_)
print("最佳交叉验证MSE: {:.4f}".format(-grid_search.best_score_))

# 使用最佳模型进行预测
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_reg_scaled)

# 4. 模型评估
print("\n=== 回归模型评估 ===")

# 计算评价指标
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
mae = mean_absolute_error(y_test_reg, y_pred_rf)
r2 = r2_score(y_test_reg, y_pred_rf)

print("测试集RMSE: {:.4f}".format(rmse))
print("测试集MAE: {:.4f}".format(mae))
print("测试集R2: {:.4f}".format(r2))

# 5. 95%预测区间
errors = y_test_reg - y_pred_rf
ci_95 = 1.96 * np.std(errors)
coverage = np.mean((y_test_reg >= (y_pred_rf - ci_95)) & 
                  (y_test_reg <= (y_pred_rf + ci_95)))
print(f"95%预测区间宽度: {ci_95:.4f}")
print(f"覆盖率: {coverage:.4f}")

# 6. 按浓度分组的误差分析
concentrations = sorted(y_test_reg.unique())
print("\n按浓度分组的MAE:")
for conc in concentrations:
    mask = (y_test_reg == conc)
    if np.sum(mask) > 0:
        mae = mean_absolute_error(y_test_reg[mask], y_pred_rf[mask])
        print(f"浓度{conc}ppm的MAE: {mae:.4f}")

# 7. 特征重要性分析
print("\n=== 特征重要性分析 ===")
feature_importance = pd.DataFrame({
    '特征': X_reg.columns,
    '重要性': best_rf.feature_importances_
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)
print(feature_importance)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='重要性', y='特征')
plt.title('浓度回归模型特征重要性分析', fontsize=14, fontweight='bold')
plt.xlabel('重要性', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'concentration_feature_importance.png'), dpi=300, bbox_inches='tight')
# plt.show()

# 8. 5折交叉验证分析
print("\n=== 5折交叉验证分析 ===")
from sklearn.model_selection import cross_validate, KFold

def plot_cross_validation_results(estimator, X, y, cv=None, n_jobs=-1):
    plt.figure(figsize=(12, 10))
    
    # 执行5折交叉验证，获取多个指标
    cv_results = cross_validate(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=True
    )
    
    # 获取每次折叠的结果
    train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'])
    test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])
    train_mae = -cv_results['train_neg_mean_absolute_error']
    test_mae = -cv_results['test_neg_mean_absolute_error']
    train_r2 = cv_results['train_r2']
    test_r2 = cv_results['test_r2']
    
    # 创建折叠编号
    fold_numbers = np.arange(1, len(train_rmse) + 1)
    
    # 绘制RMSE结果
    plt.subplot(3, 1, 1)
    plt.plot(fold_numbers, train_rmse, 'o-', color='r', label='训练集RMSE')
    plt.plot(fold_numbers, test_rmse, 'o-', color='g', label='验证集RMSE')
    
    # 添加数值标注（保留4位小数）
    for i, (train_r, test_r) in enumerate(zip(train_rmse, test_rmse)):
        plt.annotate(f'{train_r:.4f}', (fold_numbers[i], train_r), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{test_r:.4f}', (fold_numbers[i], test_r), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.xlabel('折叠编号', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('5折交叉验证 - RMSE', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fold_numbers, fold_numbers)
    
    # 绘制MAE结果
    plt.subplot(3, 1, 2)
    plt.plot(fold_numbers, train_mae, 'o-', color='r', label='训练集MAE')
    plt.plot(fold_numbers, test_mae, 'o-', color='g', label='验证集MAE')
    
    # 添加数值标注（保留4位小数）
    for i, (train_m, test_m) in enumerate(zip(train_mae, test_mae)):
        plt.annotate(f'{train_m:.4f}', (fold_numbers[i], train_m), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{test_m:.4f}', (fold_numbers[i], test_m), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.xlabel('折叠编号', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('5折交叉验证 - MAE', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fold_numbers, fold_numbers)
    
    # 绘制R2结果
    plt.subplot(3, 1, 3)
    plt.plot(fold_numbers, train_r2, 'o-', color='r', label='训练集R2')
    plt.plot(fold_numbers, test_r2, 'o-', color='g', label='验证集R2')
    
    # 添加数值标注（保留4位小数）
    for i, (train_r2_val, test_r2_val) in enumerate(zip(train_r2, test_r2)):
        plt.annotate(f'{train_r2_val:.4f}', (fold_numbers[i], train_r2_val), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{test_r2_val:.4f}', (fold_numbers[i], test_r2_val), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
    
    plt.xlabel('折叠编号', fontsize=12)
    plt.ylabel('R2', fontsize=12)
    plt.title('5折交叉验证 - R2', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fold_numbers, fold_numbers)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印交叉验证结果
    print("\n5折交叉验证结果:")
    print("折叠编号:", fold_numbers)
    print("训练集RMSE:", [f'{rmse:.4f}' for rmse in train_rmse])
    print("验证集RMSE:", [f'{rmse:.4f}' for rmse in test_rmse])
    print("训练集MAE:", [f'{mae:.4f}' for mae in train_mae])
    print("验证集MAE:", [f'{mae:.4f}' for mae in test_mae])
    print("训练集R2:", [f'{r2:.4f}' for r2 in train_r2])
    print("验证集R2:", [f'{r2:.4f}' for r2 in test_r2])
    
    # 计算并打印统计信息
    print("\n统计信息:")
    print(f"验证集RMSE - 平均值: {np.mean(test_rmse):.4f}, 标准差: {np.std(test_rmse):.4f}")
    print(f"验证集MAE - 平均值: {np.mean(test_mae):.4f}, 标准差: {np.std(test_mae):.4f}")
    print(f"验证集R2 - 平均值: {np.mean(test_r2):.4f}, 标准差: {np.std(test_r2):.4f}")
    
    # 保存交叉验证数据到文件
    cv_data = pd.DataFrame({
        '折叠编号': fold_numbers,
        '训练集RMSE': train_rmse,
        '验证集RMSE': test_rmse,
        '训练集MAE': train_mae,
        '验证集MAE': test_mae,
        '训练集R2': train_r2,
        '验证集R2': test_r2
    })
    
    # 保存为CSV文件
    csv_path = os.path.join(OUTPUT_DIR, 'cross_validation_data.csv')
    cv_data.to_csv(csv_path, index=False, encoding='gbk')
    print(f"\n交叉验证数据已保存到CSV文件: {csv_path}")
    
    # 保存为Excel文件
    excel_path = os.path.join(OUTPUT_DIR, 'cross_validation_data.xlsx')
    cv_data.to_excel(excel_path, index=False)
    print(f"交叉验证数据已保存到Excel文件: {excel_path}")
    
    return cv_results

# 执行5折交叉验证
print("正在执行5折交叉验证...")
cv_results = plot_cross_validation_results(
    best_rf, X_train_reg_scaled, y_train_reg, 
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)



# 预测值 vs 真实值散点图
print("\n=== 预测值 vs 真实值可视化 ===")

plt.figure(figsize=(10, 8))
plt.scatter(y_test_reg, y_pred_rf, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('真实浓度 (ppm)', fontsize=12)
plt.ylabel('预测浓度 (ppm)', fontsize=12)
plt.title('浓度回归模型预测值 vs 真实值', fontsize=14, fontweight='bold')

# 修改文本显示方式，避免使用上标字符
r2_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.4f}'
plt.text(0.05, 0.95, r2_text,
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'concentration_prediction_scatter.png'), dpi=300, bbox_inches='tight')
# plt.show()

# 模型保存
model_path = os.path.join(OUTPUT_DIR, 'best_rf_model.joblib')
dump(best_rf, model_path)
print(f"\n最佳模型已保存到: {model_path}")

# 保存结果到文件
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

# 保存为JSON文件
import json
results_json_path = os.path.join(OUTPUT_DIR, 'model_results.json')
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"\n模型结果已保存到JSON文件: {results_json_path}")

print("\n浓度回归模型分析完成！")
