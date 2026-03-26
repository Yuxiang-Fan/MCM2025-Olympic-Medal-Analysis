import pandas as pd
import numpy as np

def load_task2_data():
    """
    加载任务2所需的社会经济与赛事特征数据
    返回历史仅获1枚奖牌的国家数据和历史零奖牌的国家数据
    在实际工程中此处应替换为真实的数据读取路径
    """
    cols = ['Country', 'alpha', 'beta', 'gamma', 'delta', 'epsilon']
    
    # 构造历史仅获1枚奖牌的国家示例数据
    # alpha代表政治稳定性 beta代表GDP指标 gamma基于1减去HHI计算 delta等于1除以参赛国家数 epsilon代表运动员综合实力
    one_medal_data = [
        ['Country_A', 1, 0.5, 0.8, 0.2, 1],
        ['Country_B', 0, 1.0, 0.6, 0.1, 0],
        ['Country_C', 1, 0.0, 0.7, 0.15, 1]
    ]
    one_medal_df = pd.DataFrame(one_medal_data, columns=cols)
    
    # 构造零奖牌目标国家示例数据
    zero_medal_data = [
        ['Andorra', 1, 0.5, 0.5, 0.1, 0],
        ['Angola', 0, 0.0, 0.3, 0.05, 0]
    ]
    zero_medal_df = pd.DataFrame(zero_medal_data, columns=cols)
    
    return one_medal_df, zero_medal_df

def calculate_comprehensive_score(row):
    """
    根据各项指标线性加权计算国家综合奥运实力Score
    权重分配基于先验知识或AHP层次分析法设定
    """
    score = (0.2 * row['alpha'] + 
             0.2 * row['beta'] + 
             0.15 * row['gamma'] + 
             0.15 * row['delta'] + 
             0.3 * row['epsilon'])
    return score

def monte_carlo_simulation(base_prob, n_simulations=10000):
    """
    使用蒙特卡洛方法模拟目标国家夺得首枚奖牌的真实概率
    通过大量伯努利试验将连续的概率得分映射为离散赛果的统计频率
    """
    if pd.isna(base_prob):
        return np.nan
        
    # 限制概率边界防止溢出
    base_prob = min(max(base_prob, 0), 1)
    
    # 生成随机数矩阵并统计成功夺牌的总次数
    wins = np.sum(np.random.rand(n_simulations) < base_prob)
    
    # 计算夺牌频率作为最终概率
    return wins / n_simulations

def main():
    # 固定随机种子保证蒙特卡洛模拟结果可稳定复现
    np.random.seed(42)
    
    one_medal_df, zero_medal_df = load_task2_data()

    # 1. 求解基准阈值T
    one_medal_df['Score'] = one_medal_df.apply(calculate_comprehensive_score, axis=1)
    threshold_T = one_medal_df['Score'].mean()
    print(f"基准阈值T标定完成: {threshold_T:.4f}")
    
    if threshold_T == 0:
        print("致命错误：基准阈值为0，无法进行概率归一化计算")
        return
    
    # 2. 计算零奖牌国家的Score与初始概率
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
