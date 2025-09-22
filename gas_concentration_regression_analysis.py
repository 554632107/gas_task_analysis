"""
气体浓度回归分析脚本
核心任务：针对每种气体，进行浓度值回归预测分析、置信区间计算和模型评估
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'gas_regression_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    """加载数据，仅尝试常用编码"""
    print("=> 正在加载数据...")
    try:
        # 优先尝试utf-8编码
        df = pd.read_csv(file_path, encoding='utf-8')
        print("使用utf-8编码成功加载数据")
    except:
        try:
            # 备用尝试gbk编码
            df = pd.read_csv(file_path, encoding='gbk')
            print("使用gbk编码成功加载数据")
        except Exception as e:
            raise ValueError(f"无法读取文件 {file_path}: {str(e)}")
    
    # 验证必要列是否存在
    required_columns = ['气体种类', '气体浓度(ppm)', '测量结果', '气体谱线(nm)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件缺少必要列: {', '.join(missing_cols)}")
    
    # 提取唯一气体种类和浓度值
    gas_types = df['气体种类'].unique()
    concentrations = sorted(df['气体浓度(ppm)'].unique())
    
    print(f"检测到 {len(gas_types)} 种气体类型: {', '.join(gas_types)}")
    print(f"检测到 {len(concentrations)} 个浓度级别: {', '.join(map(str, concentrations))}")
    
    return df, gas_types, concentrations

def calculate_confidence_interval(scores, confidence=0.95):
    """计算置信区间"""
    if len(scores) < 2:
        return np.nan, np.nan
    
    mean_score = np.mean(scores)
    std_err = stats.sem(scores)
    
    # 计算t分布临界值
    t_critical = stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    margin_of_error = t_critical * std_err
    
    ci_lower = max(0, mean_score - margin_of_error)
    ci_upper = mean_score + margin_of_error
    
    return ci_lower, ci_upper



def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, gas_type):
    """评估回归模型并返回关键指标"""
    print(f"   => 评估 {model_name} 模型...")
    
    # 训练和预测
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 计算基本指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 交叉验证计算MAE置信区间
    try:
        cv_results = cross_validate(model, X_train, y_train, cv=5, 
                                  scoring='neg_mean_absolute_error')
        mae_scores = -cv_results['test_score']
        mae_ci_lower, mae_ci_upper = calculate_confidence_interval(mae_scores)
    except Exception as e:
        print(f"   ! 交叉验证出错: {str(e)}")
        mae_ci_lower, mae_ci_upper = np.nan, np.nan
    
    # 计算95%预测区间覆盖率
    errors = y_test - y_pred
    ci_95 = 1.96 * np.std(errors)
    coverage = np.mean((y_test >= (y_pred - ci_95)) & (y_test <= (y_pred + ci_95)))
    
    # 打印关键结果
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE 95% CI: [{mae_ci_lower:.4f}, {mae_ci_upper:.4f}]")
    print(f"95%预测区间覆盖率: {coverage:.4f}")
    
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
    """针对单种气体进行浓度回归分析"""
    print(f"\n=> 开始分析气体: {gas_type}")
    
    # 提取该气体的数据
    gas_mask = (df['气体种类'] == gas_type)
    X_gas = X[gas_mask]
    y_gas = y[gas_mask]
    
    # 检查数据量
    if len(X_gas) < 10:
        print(f"   ! 警告: {gas_type} 数据量不足 ({len(X_gas)} 条)，跳过分析")
        return None
    
    print(f"获取 {gas_type} 数据: {len(X_gas)} 条记录")
    

    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_gas, y_gas, test_size=0.2, random_state=42
    )
    
    # 初始化模型
    lr_reg = LinearRegression()
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 评估模型
    lr_results = evaluate_model(lr_reg, X_train, X_test, y_train, y_test, "线性回归", gas_type)
    rf_results = evaluate_model(rf_reg, X_train, X_test, y_train, y_test, "随机森林", gas_type)
    
    # 保存结果
    results = pd.DataFrame([
        {'模型': '线性回归', **lr_results},
        {'模型': '随机森林', **rf_results}
    ])
    
    # 生成关键预测图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_reg.predict(X_test), alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('真实浓度(ppm)')
    plt.ylabel('预测浓度(ppm)')
    plt.title(f'{gas_type} 随机森林预测效果')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{gas_type}_prediction.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存 {gas_type} 预测结果图表")
    
    return results

def main():
    print("===== 气体浓度回归分析任务开始 =====")
    
    # 1. 加载数据
    data_file = os.path.join(PROJECT_ROOT, 'gas_dataset.csv')
    try:
        df, gas_types, concentrations = load_data(data_file)
        print("\n=> 数据加载完成")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return
    
    # 2. 数据预处理
    print("\n=> 开始数据预处理...")
    X = df[['测量结果', '气体谱线(nm)']]
    y = df['气体浓度(ppm)'].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("特征标准化完成")
    
    # 3. 按气体类型分析
    all_results = []
    for gas_type in gas_types:
        results = analyze_gas_type(df, X_scaled, y, gas_type)
        if results is not None:
            all_results.append((gas_type, results))
    
    if not all_results:
        print("\n! 所有气体分析均失败，任务终止")
        return
    
    # 4. 汇总结果
    print("\n=> 生成汇总结果...")
    summary_data = []
    for gas_type, results in all_results:
        for _, row in results.iterrows():
            row_dict = row.to_dict()
            row_dict['气体类型'] = gas_type
            summary_data.append(row_dict)
    
    summary = pd.DataFrame(summary_data)
    
    # 保存汇总结果
    summary.to_csv(os.path.join(OUTPUT_DIR, 'regression_summary.csv'), 
                  index=False, encoding='utf-8-sig')
    print("汇总结果已保存至 regression_summary.csv")
    
    # 5. 生成关键图表
    print("\n=> 生成关键分析图表...")
    
    # MAE比较图
    plt.figure(figsize=(10, 6))
    for gas_type in gas_types:
        gas_data = summary[summary['气体类型'] == gas_type]
        for model in ['线性回归', '随机森林']:
            model_data = gas_data[gas_data['模型'] == model]
            mae = model_data['MAE'].values[0]
            plt.bar(f'{model}\n{gas_type}', mae, alpha=0.7)
    
    plt.title('MAE比较')
    plt.ylabel('MAE (ppm)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    

    
    print("关键图表已生成")
    
    # 6. 打印最终总结
    print("\n===== 分析结果总结 =====")
    print(f"成功分析 {len(all_results)} 种气体类型")
    
    # 按气体类型显示最佳模型
    for gas_type in gas_types:
        gas_data = summary[summary['气体类型'] == gas_type]
        if not gas_data.empty:
            best_model = gas_data.loc[gas_data['MAE'].idxmin(), '模型']
            best_mae = gas_data['MAE'].min()
            print(f"- {gas_type}: 最佳模型为 {best_model} (MAE={best_mae:.4f})")
    
    print("\n===== 气体浓度回归分析任务完成 =====")
    print(f"结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
