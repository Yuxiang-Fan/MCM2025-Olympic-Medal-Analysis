import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr, ttest_ind
import warnings
warnings.filterwarnings('ignore')

def load_task3_data():
    """
    [数据加载占位]
    期望维度: (N_样本, M_特征) -> 一行代表某国在某届奥运会的表现
    
    必须包含的列 (Schema):
    - 'Country': 国家名称 (str)
    - 'Year': 奥运会年份 (int)
    - 'Total_Medals': 该国当届获得的奖牌总数 (y_i)
    - 'Host': 是否为当届东道主 (1 为是，0 为否)
    - [赛事列1], [赛事列2] ...: 如 'Archery', 'Athletics' 等，代表该国在该项目获得的奖牌数 (x_ij)
    """
    df = pd.DataFrame()
    # 实际运行时，请替换为: df = pd.read_csv('your_data_path.csv')
    return df

def check_and_remove_vif(X, threshold=10.0):
    """
    计算方差膨胀因子 (VIF)，剔除多重共线性较高的赛事特征
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_scaled_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled_df.values, i) 
                       for i in range(len(X_scaled_df.columns))]
    
    valid_features = vif_data[vif_data["VIF"] < threshold]["Feature"].tolist()
    return X[valid_features], vif_data

def run_lasso_regression(X, y):
    """
    Lasso 回归模型，最小化目标函数提取各项目系数
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
    
    coefficients = pd.DataFrame({
        'Sports Events': X.columns,
        'Lasso Regression Coefficient': lasso.coef_
    })
    
    coefficients['Abs_Coef'] = coefficients['Lasso Regression Coefficient'].abs()
    coefficients = coefficients.sort_values(by='Abs_Coef', ascending=False).drop(columns=['Abs_Coef'])
    
    return coefficients

def analyze_host_effect(df):
    """
    探究东道主信息造成的奖牌数变化
    """
    # 1. Pearson 相关系数计算
    host_info = df['Host']
    y = df['Total_Medals']
    correlation, pearson_p = pearsonr(host_info, y)
    print(f"Pearson 相关系数: {correlation:.4f} (p-value: {pearson_p:.2e})")
    
    # 2. t-test 显著性检验
    host_1 = df[df['Host'] == 1]['Total_Medals']
    host_0 = df[df['Host'] == 0]['Total_Medals']
    t_stat, t_p_value = ttest_ind(host_1, host_0, equal_var=False)
    print(f"东道主与非东道主总奖牌 t-test p-value: {t_p_value:.2e}")
    
    # 3. 奖牌变化密度 (Medal Variation Density) 计算
    df_sorted = df.sort_values(by=['Country', 'Year'])
    df_sorted['Medals_Change_Density'] = (df_sorted.groupby('Country')['Total_Medals'].diff().abs() / df_sorted['Total_Medals']) * 100
    
    # 根据规则对事件进行分类
    def classify_event(x):
        if pd.isna(x):
            return 'Unknown'
        elif x > 10:
            return 'Dominant Event'
        elif x > 5:
            return 'Neutral Event'
        else:
            return 'Minor Event'
            
    df_sorted['Event_Classification'] = df_sorted['Medals_Change_Density'].apply(classify_event)
    return df_sorted

def main():
    df = load_task3_data()
    
    if df.empty:
        print("提示：检测到空数据集。请查阅 README 接入真实数据后运行。")
        return

    # 从 DataFrame 中分离目标变量和赛事特征矩阵
    y = df['Total_Medals']
    # 假设除了 Country, Year, Total_Medals, Host 外，其余均为赛事列
    non_event_cols = ['Country', 'Year', 'Total_Medals', 'Host']
    X = df.drop(columns=[col for col in non_event_cols if col in df.columns])
    
    print("--- 1. VIF 共线性检验 ---")
    X_valid, vif_res = check_and_remove_vif(X)
    
    print("\n--- 2. Lasso 回归项目重要性提取 ---")
    lasso_coefs = run_lasso_regression(X_valid, y)
    print("重要性前 10 的体育项目:")
    print(lasso_coefs.head(10))
    
    print("\n--- 3. 东道主效应深度分析 ---")
    density_df = analyze_host_effect(df)
    print("\n奖牌变化密度及分类提取完毕。")

if __name__ == "__main__":
    main()