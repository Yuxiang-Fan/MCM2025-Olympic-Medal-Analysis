import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr, ttest_ind

warnings.filterwarnings('ignore')

def load_task3_data():
    """
    构造包含多重共线性的模拟数据，确保脚本可独立运行验证
    实际工程中应替换为真实的赛事数据
    """
    np.random.seed(42)
    n_samples = 150
    
    data = {
        'Country': ['Country_' + str(i % 15) for i in range(n_samples)],
        'Year': [2000 + (i % 6) * 4 for i in range(n_samples)],
        'Total_Medals': np.random.poisson(lam=20, size=n_samples),
        'Host': np.random.binomial(n=1, p=0.05, size=n_samples),
        'Athletics': np.random.poisson(lam=5, size=n_samples),
        'Swimming': np.random.poisson(lam=4, size=n_samples),
        'Gymnastics': np.random.poisson(lam=3, size=n_samples),
        'Shooting': np.random.poisson(lam=2, size=n_samples)
    }
    
    # 刻意制造高度共线性的特征：假设跳水奖牌数与游泳高度绑定
    data['Diving'] = [x + np.random.poisson(lam=1) for x in data['Swimming']]
    
    return pd.DataFrame(data)

def filter_features_by_vif(X, threshold=10.0):
    """
    计算VIF并剔除多重共线性严重的特征，防止回归系数失真
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    
    # 保留VIF处于安全阈值内的特征
    valid_features = vif_data[vif_data["VIF"] < threshold]["Feature"].tolist()
    return X[valid_features], vif_data

def extract_lasso_coefs(X, y):
    """
    利用带交叉验证的Lasso回归进行特征降维，提取对总奖牌贡献最大的核心项目
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LassoCV自动在正则化路径上寻找最优的惩罚系数
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_scaled, y)
    
    coef_df = pd.DataFrame({
        'Sport': X.columns,
        'Coefficient': lasso.coef_
    })
    
    # 剔除被Lasso压缩为0的无关特征，并按绝对值降序
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df[coef_df['Abs_Coef'] > 0].sort_values(by='Abs_Coef', ascending=False)
    coef_df = coef_df.drop(columns=['Abs_Coef'])
    
    return coef_df

def analyze_host_effect(df):
    """
    从相关性和均值差异两个维度验证东道主效应
    """
    # 1. Pearson相关性分析
    corr, p_pearson = pearsonr(df['Host'], df['Total_Medals'])
    print(f"Host与总奖牌的Pearson相关系数: {corr:.4f} (p-value: {p_pearson:.2e})")
    
    # 2. 独立样本t检验 (不假设方差齐性)
    host_medals = df[df['Host'] == 1]['Total_Medals']
    non_host_medals = df[df['Host'] == 0]['Total_Medals']
    t_stat, p_t = ttest_ind(host_medals, non_host_medals, equal_var=False)
    print(f"东道主效应独立样本t检验 p-value: {p_t:.2e}")
    
    # 3. 奖牌波动率计算
    df_sorted = df.sort_values(by=['Country', 'Year']).copy()
    # 避免除以0的警告
    medals_denom = df_sorted['Total_Medals'].replace(0, np.nan)
    df_sorted['Medals_Change_Density'] = (df_sorted.groupby('Country')['Total_Medals'].diff().abs() / medals_denom) * 100
    
    def classify_event(val):
        if pd.isna(val):
            return 'Unknown'
        elif val > 10:
            return 'Dominant'
        elif val > 5:
            return 'Neutral'
        else:
            return 'Minor'
            
    df_sorted['Event_Class'] = df_sorted['Medals_Change_Density'].apply(classify_event)
    return df_sorted

def main():
    df = load_task3_data()
    
    y = df['Total_Medals']
    # 分离非赛事类特征，留下纯体育项目矩阵
    meta_cols = ['Country', 'Year', 'Total_Medals', 'Host']
    X = df.drop(columns=[col for col in meta_cols if col in df.columns])
    
    print(">>> 1. 执行VIF共线性诊断")
    X_valid, vif_res = filter_features_by_vif(X)
    print("剔除的高共线性特征:", set(X.columns) - set(X_valid.columns))
    
    print("\n>>> 2. Lasso特征选择结果")
    lasso_coefs = extract_lasso_coefs(X_valid, y)
    print("对总奖牌贡献显著的体育项目:\n", lasso_coefs.head(10))
    
    print("\n>>> 3. 东道主效应统计检验")
    density_df = analyze_host_effect(df)
    print("\n[状态] 任务3多重共线性处理与效应检验完成。")

if __name__ == "__main__":
    main()
