import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

def generate_mock_medal_data():
    """
    构造模拟的奥运奖牌明细数据，确保脚本可直接运行验证逻辑
    包含年份、国家、项目及奖牌类型
    """
    np.random.seed(42)
    years = [2004, 2008, 2012, 2016, 2020, 2024]
    countries = ['China', 'United States', 'Cuba', 'Japan', 'Italy', 'Brazil']
    sports = ['Volleyball', 'Swimming', 'Athletics']
    
    data = []
    for y in years:
        for c in countries:
            for s in sports:
                # 随机生成奖牌，模拟女排项目在中美执教前后的表现变化
                medal = 'None'
                prob = np.random.rand()
                if s == 'Volleyball':
                    # 模拟郎平执教后的效应提升
                    if (c == 'United States' and 2008 < y <= 2012) or (c == 'China' and y >= 2016):
                        prob += 0.4
                
                if prob > 0.9: medal = 'Gold'
                elif prob > 0.7: medal = 'Silver'
                elif prob > 0.5: medal = 'Bronze'
                
                data.append([y, c, s, medal])
                
    return pd.DataFrame(data, columns=['Year', 'Country', 'Sport', 'Medal'])

def build_did_features(df):
    """
    量化奖牌得分并构造DID回归所需的交互项
    """
    # 奖牌权重赋值：金3银2铜1
    m_map = {'Gold': 3, 'Silver': 2, 'Bronze': 1, 'None': 0}
    df['Medal_Score'] = df['Medal'].map(m_map).fillna(0)
    
    # 构造实验组标识 Treat
    df['Is_USA'] = (df['Country'] == 'United States').astype(int)
    df['Is_CHN'] = (df['Country'] == 'China').astype(int)
    
    # 构造政策发生时间标识 Post
    # 设定美国队效应期为2008年后，中国队为2016年后
    df['After_Coach_USA'] = ((df['Year'] > 2008) & (df['Country'] == 'United States')).astype(int)
    df['After_Coach_CHN'] = ((df['Year'] >= 2016) & (df['Country'] == 'China')).astype(int)
    
    return df

def fit_did_impact(df):
    """
    使用OLS拟合双重差分模型，提取名帅执教的净效应系数
    """
    vball_data = df[df['Sport'] == 'Volleyball'].copy()
    
    if vball_data.empty:
        return None
        
    # 定义DID模型公式：包含国家与年份的固定效应
    # 核心关注交叉项 Is_USA:After_Coach_USA 的系数
    formula = "Medal_Score ~ Is_USA:After_Coach_USA + Is_CHN:After_Coach_CHN + C(Year) + C(Country)"
    
    res = smf.ols(formula, data=vball_data).fit()
    
    # 提取评估系数与显著性P值
    res_summary = {
        'USA_Effect': res.params.get('Is_USA:After_Coach_USA', 0),
        'USA_P': res.pvalues.get('Is_USA:After_Coach_USA', 1),
        'CHN_Effect': res.params.get('Is_CHN:After_Coach_CHN', 0),
        'CHN_P': res.pvalues.get('Is_CHN:After_Coach_CHN', 1)
    }
    
    print("\n[DID模型估计结果]")
    print(f"美国队执教增益: {res_summary['USA_Effect']:.3f} (P={res_summary['USA_P']:.4f})")
    print(f"中国队执教增益: {res_summary['CHN_Effect']:.3f} (P={res_summary['CHN_P']:.4f})")
    
    return res_summary

def find_investment_targets(df, target_list):
    """
    筛选曾有辉煌历史但近期表现断档的项目，作为名帅引进的潜力目标
    """
    targets = []
    for country in target_list:
        c_df = df[df['Country'] == country]
        for sport in c_df['Sport'].unique():
            s_df = c_df[c_df['Sport'] == sport]
            # 对比2016年前后的表现差
            pre_val = s_df[s_df['Year'] < 2016]['Medal_Score'].sum()
            post_val = s_df[s_df['Year'] >= 2016]['Medal_Score'].sum()
            
            if pre_val >= 2 and post_val == 0:
                targets.append({'Country': country, 'Sport': sport, 'Gap': pre_val})
                
    return pd.DataFrame(targets)

def eval_effect_transfer(df, target_country):
    """
    利用Cosine Similarity计算目标国家与中美队伍的战绩相似度
    以此作为名帅效应迁移的参考权重
    """
    # 提取各年度奖牌得分作为特征向量
    pivot_df = df.pivot_table(index='Country', columns='Year', values='Medal_Score', aggfunc='sum').fillna(0)
    
    if target_country not in pivot_df.index:
        return 0
        
    vec_target = pivot_df.loc[[target_country]].values
    vec_china = pivot_df.loc[['China']].values
    
    # 计算余弦相似度
    sim = cosine_similarity(vec_target, vec_china)[0][0]
    print(f"\n{target_country} 与中国队的历史战绩余弦相似度: {sim:.4f}")
    return sim

def main():
    # 数据加载
    df = generate_mock_medal_data()
    
    # 特征工程
    df = build_did_features(df)
    
    # 任务4核心：DID因果推断
    did_results = fit_did_impact(df)
    
    # 寻找潜力投资项目
    potential_df = find_investment_targets(df, ['Cuba', 'Japan', 'Italy'])
    if not potential_df.empty:
        print("\n建议引入名帅的潜力方向:\n", potential_df)
        
    # 效应迁移分析
    eval_effect_transfer(df, "Cuba")

if __name__ == "__main__":
    main()
