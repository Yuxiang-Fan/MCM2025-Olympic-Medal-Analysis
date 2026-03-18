import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_task4_data():
    """
    [数据加载占位]
    维度: (N_样本, 4) -> 每一行代表某国在某年某项目上的获奖情况
    
    必须包含的列 (Schema):
    - 'Year': 奥运会年份 (int)
    - 'Country': 国家名称 (str)
    - 'Sport': 体育项目名称，如 'Volleyball', 'Baseball' 等 (str)
    - 'Medal': 奖牌类型，取值为 'Gold', 'Silver', 'Bronze' 或 'None'/'NaN'
    """
    df = pd.DataFrame(columns=['Year', 'Country', 'Sport', 'Medal'])
    # 实际运行时，请替换为: df = pd.read_csv('your_medal_details.csv')
    return df

def preprocess_did_data(df):
    """
    数据预处理与 DID 哑变量构造
    """
    # 1. 奖牌量化赋值 (Gold=3, Silver=2, Bronze=1, 无奖牌=0)
    medal_map = {'Gold': 3, 'Silver': 2, 'Bronze': 1, 'None': 0}
    df['Medal_num'] = df['Medal'].map(medal_map).fillna(0)
    
    # 2. 构造双重差分 (DID) 需要的哑变量 (Dummy Variables)
    # Treat 变量：是否为干预组
    df['Treat_USA'] = (df['Country'] == 'United States').astype(int)
    df['Treat_China'] = (df['Country'] == 'China').astype(int)
    
    # Post 变量：干预发生后的时间截面
    # 郎平 2008 年后执教美国队，2016 年起执教中国队
    df['Post_USA'] = ((df['Year'] > 2008) & (df['Country'] == 'United States')).astype(int)
    df['Post_China'] = ((df['Year'] >= 2016) & (df['Country'] == 'China')).astype(int)
    
    return df

def run_did_model(df):
    """
    运行双重差分模型评估郎平执教效应
    """
    # 筛选女排赛事数据进行模型拟合
    volleyball_df = df[df['Sport'] == 'Volleyball'].copy()
    
    if volleyball_df.empty:
        return None, None
        
    # DID 回归公式 (结合国家和年份固定效应)
    formula = "Medal_num ~ C(Treat_USA):C(Post_USA) + C(Treat_China):C(Post_China) + C(Year) + C(Country)"
    
    # 使用 OLS 拟合模型
    model = smf.ols(formula, data=volleyball_df).fit()
    
    # 提取交叉项系数 (即名帅效应带来的额外奖牌数量)
    beta_usa = model.params.get('C(Treat_USA)[T.1]:C(Post_USA)[T.1]', np.nan)
    p_value_usa = model.pvalues.get('C(Treat_USA)[T.1]:C(Post_USA)[T.1]', np.nan)
    
    beta_china = model.params.get('C(Treat_China)[T.1]:C(Post_China)[T.1]', np.nan)
    p_value_china = model.pvalues.get('C(Treat_China)[T.1]:C(Post_China)[T.1]', np.nan)
    
    print("\n--- 郎平执教效应 (DID 评估结果) ---")
    print(f"对美国队的影响系数 (beta_3): {beta_usa:.4f}, p-value: {p_value_usa:.4f}")
    print(f"对中国队的影响系数 (beta_6): {beta_china:.4f}, p-value: {p_value_china:.4f}")
    
    return beta_usa, beta_china

def identify_potential_projects(df, target_countries):
    """
    寻找目标国家(如古巴、日本、意大利)需要引入名帅的潜力项目
    判定标准：2016年以前获得过至少2枚奖牌，但2016年及以后获得0枚奖牌
    """
    potential_projects = []
    
    for country in target_countries:
        country_df = df[df['Country'] == country]
        if country_df.empty:
            continue
            
        sports = country_df['Sport'].unique()
        for sport in sports:
            sport_df = country_df[country_df['Sport'] == sport]
            
            # 2016年前的奖牌数
            pre_2016_medals = sport_df[sport_df['Year'] < 2016]['Medal_num'].sum()
            # 2016年及以后的奖牌数
            post_2016_medals = sport_df[sport_df['Year'] >= 2016]['Medal_num'].sum()
            
            if pre_2016_medals >= 2 and post_2016_medals == 0:
                potential_projects.append({'Country': country, 'Sport': sport})
                
    return pd.DataFrame(potential_projects)

def calculate_similarity_and_transfer(df, target_country, pre_coach_years):
    """
    扩展报告 7.4：使用余弦相似度进行效应迁移评估
    通过对比目标国家与中美在名帅执教前5届的战绩相似度，来决定效应迁移的权重
    """
    # 占位：实际需提取向量并执行 cosine_similarity([vec_target], [vec_source])
    # 这里仅为展示计算逻辑框架
    print(f"正在计算 {target_country} 与中美队伍的余弦相似度以分配名帅效应权重...")
    # 假设计算得出的相似度权重矩阵
    similarity_weight = 0.85 
    return similarity_weight

def main():
    df = load_task4_data()
    
    if df.empty:
        print("提示：检测到空数据集。请查阅 README 接入真实明细数据后运行。")
        return
        
    print("--- 1. 数据预处理与变量构造 ---")
    df = preprocess_did_data(df)
    
    print("\n--- 2. 双重差分 (DID) 模型拟合 ---")
    beta_usa, beta_china = run_did_model(df)
    
    print("\n--- 3. 寻找急需名帅投资的潜力项目 ---")
    target_countries = ['Cuba', 'Japan', 'Italy']
    projects_df = identify_potential_projects(df, target_countries)
    
    if not projects_df.empty:
        print(f"筛选出符合标准（曾有辉煌但近期断档）的潜力项目：\n{projects_df}")
    else:
        print("（由于接入的是空数据集，暂无匹配的潜力项目输出）")
        
    print("\n--- 4. (报告 7.4) 基于余弦相似度的效应迁移评估 ---")
    # 模拟对阿尔及利亚或阿根廷等目标国家进行相似度计算
    calculate_similarity_and_transfer(df, "Argentina", pre_coach_years=5)

if __name__ == "__main__":
    main()