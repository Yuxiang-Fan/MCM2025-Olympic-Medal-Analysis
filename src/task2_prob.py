import pandas as pd
import numpy as np

def load_task2_data():
    """
    [数据加载占位]
    """
    
    # 1. 历史上仅获得过 1 枚奖牌的国家数据 (用于标定基准阈值 T)
    # 期望维度: (N_1, 6) -> N_1 为符合条件的国家数量
    # 列说明:
    #   'Country': 国家名称 (str)
    #   'alpha': 政治稳定性指标 (取值: 0 或 1)
    #   'beta': GDP 指标 (取值: 0, 0.5 或 1)
    #   'gamma': 最优势项目奖牌分配程度 (计算公式: 1 - HHI)
    #   'delta': 最优势项目竞争程度 (计算公式: 1 / 参赛国家数)
    #   'epsilon': 运动员综合实力 (取值: 0 或 1)
    one_medal_countries = pd.DataFrame(columns=[
        'Country', 'alpha', 'beta', 'gamma', 'delta', 'epsilon'
    ])
    
    # 2. 历史上从未获得过奖牌的国家数据 (预测目标，如 Andorra, Angola 等)
    # 期望维度: (N_0, 6) -> N_0 为零奖牌国家数量 (报告中提及超过60个)
    zero_medal_countries = pd.DataFrame(columns=[
        'Country', 'alpha', 'beta', 'gamma', 'delta', 'epsilon'
    ])
    
    return one_medal_countries, zero_medal_countries

def calculate_comprehensive_score(row):
    """
    计算国家综合奥运实力得分 Score
    Score = 0.2*alpha + 0.2*beta + 0.15*gamma + 0.15*delta + 0.3*epsilon
    """
    score = (0.2 * row['alpha'] + 
             0.2 * row['beta'] + 
             0.15 * row['gamma'] + 
             0.15 * row['delta'] + 
             0.3 * row['epsilon'])
    return score

def monte_carlo_simulation(base_prob, n_simulations=10000):
    """
    引入蒙特卡洛方法模拟目标国家夺得首枚奖牌的概率
    """
    if pd.isna(base_prob):
        return np.nan
        
    base_prob = min(max(base_prob, 0), 1)
    # x 为模拟夺牌的次数
    x_wins = np.sum(np.random.rand(n_simulations) < base_prob)
    # p(win) = x / y
    return x_wins / n_simulations

def main():
    one_medal_df, zero_medal_df = load_task2_data()
    
    # 防止空 DataFrame 报错的占位检查
    if one_medal_df.empty or zero_medal_df.empty:
        print("提示：检测到空数据集。请查阅 README 接入真实数据后运行。")
        return

    # 1. 求解基准阈值 T
    one_medal_df['Score'] = one_medal_df.apply(calculate_comprehensive_score, axis=1)
    threshold_T = one_medal_df['Score'].mean()
    print(f"基准阈值 T 计算完成: {threshold_T:.4f}")
    
    # 2. 计算零奖牌国家的 Score 及初始概率 p_i
    zero_medal_df['Score'] = zero_medal_df.apply(calculate_comprehensive_score, axis=1)
    zero_medal_df['Base_Probability'] = zero_medal_df['Score'] / threshold_T
    
    # 3. 蒙特卡洛优化模拟
    zero_medal_df['MC_Simulated_Probability'] = zero_medal_df['Base_Probability'].apply(
        lambda p: monte_carlo_simulation(p, n_simulations=10000)
    )
    
    print("\n下届奥运会零奖牌国家首获奖牌概率预测:")
    print(zero_medal_df[['Country', 'Score', 'Base_Probability', 'MC_Simulated_Probability']])

if __name__ == "__main__":
    main()