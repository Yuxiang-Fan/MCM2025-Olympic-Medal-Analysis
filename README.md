# Olympic Medal Trend Prediction: A Multi-Model Framework (MCM 2025)

This project implements a multi-stage mathematical framework to analyze and forecast Olympic medal distributions for the **2028 Los Angeles Summer Olympics**. Developed for the **2025 MCM/ICM Problem C**, the design utilizes a combination of ensemble learning, regularization, and causal inference to evaluate the impact of socio-economic factors and strategic coaching on national athletic performance.

## Implementation Characteristics

* **Ensemble Predictive Architecture**: A stacked regressor is implemented using **Random Forest** and **Gradient Boosting (GBDT)**. The model incorporates a secondary linear regression layer to fuse non-linear predictions and generate 95% confidence intervals through self-sampling.
* **Quantified Probability Scoring**: A multi-indicator scoring system is established to evaluate the "first-medal" potential of nations with no prior Olympic success. The model integrates political stability, GDP, and project-specific competition intensity (HHI).
* **Lasso-Based Feature Selection**: Lasso regression is employed to identify the contribution coefficients of various sports disciplines. This module includes a **Variance Inflation Factor (VIF)** pre-check to mitigate multicollinearity between event settings.
* **Causal Inference of "Great Coach" Effect**: A **Difference-in-Differences (DID)** model is utilized to quantify the performance gain attributed to elite coaching (e.g., Lang Ping). The framework supports effect migration to target countries based on cosine similarity of historical performance.

## Data Requirements

To execute the provided modules, users must provide structured CSV datasets following the schemas below:

### 1. Historical Performance (Task 1)
| Feature | Type | Description |
| :--- | :--- | :--- |
| `prev_medals` | int | Total medals won by the nation in previous Olympic cycles. |
| `prev_major_medals`| int | Medals won in historically dominant disciplines. |
| `is_host` | binary | Indicator for host country status (1/0). |
| `next_events` | int | Number of events scheduled for the target Games. |

### 2. Socio-Economic Indicators (Task 2)
| Feature | Range | Description |
| :--- | :--- | :--- |
| `alpha` | {0, 1} | Binary indicator for national political stability. |
| `beta` | {0, 0.5, 1} | Categorical classification of national GDP levels. |
| `gamma` | float | Event medal allocation degree (calculated as $1 - HHI$). |
| `delta` | float | Competition intensity (calculated as $1 / n_{participants}$). |
| `epsilon`| {0, 1} | Binary indicator for comprehensive athlete strength. |

### 3. Event-Level Records (Task 3 & 4)
| Feature | Type | Description |
| :--- | :--- | :--- |
| `Year` | int | The specific Olympic year. |
| `Country` | str | Name of the participating organization. |
| `Sport` | str | The specific discipline (e.g., Athletics, Volleyball). |
| `Medal_Value` | int | Quantified score (Gold=3, Silver=2, Bronze=1, None=0). |

## Validation and Performance

The framework has been verified against historical data from 1992 to 2024 with the following observations:

* **Predictive Accuracy**: Stacking ensemble models exhibit a mean squared error (MSE) of approximately **0.043** for gold medal forecasts.
* **Statistical Significance**: The correlation between host country status and total medal count was confirmed with a $p$-value of **4.39E-278**.
* **Coaching Impact**: Analysis indicates that elite coaching contributions are statistically significant ($p < 0.05$) for target national teams.

---

# 奥运奖牌趋势预测：多模型框架 (2025 MCM)

本项目实现了一个多阶段数学框架，用于分析和预测 **2028 年洛杉矶夏季奥运会**的奖牌分布。该设计是为 **2025 年 MCM/ICM C 题**开发的，结合了集成学习、正则化和因果推断，以评估社会经济因素和战略执教对国家体育成绩的影响。

## 实现特性

* **集成预测架构**：使用 **随机森林 (Random Forest)** 和 **梯度提升决策树 (GBDT)** 实现了一个堆叠回归器（Stacked Regressor）。模型引入了二级线性回归层来融合非线性预测，并通过自采样生成 95% 的置信区间。
* **量化概率评分**：建立了一个多指标评分系统，用于评估此前未获得过奥运奖牌的国家实现“奖牌零突破”的潜力。模型综合了政治稳定性、GDP 和特定项目的竞争强度（HHI）。
* **基于 Lasso 的特征选择**：采用 Lasso 回归来识别不同体育项目的贡献系数。该模块包含 **方差膨胀因子 (VIF)** 预检，以减轻不同赛事设置之间的多重共线性问题。
* **“名师效应”的因果推断**：利用 **双重差分模型 (DID)** 来量化精英教练（如郎平）带来的成绩增益。该框架支持基于历史表现的余弦相似度，将影响评估迁移至目标国家。

## 数据要求

要运行提供的模块，用户必须提供遵循以下架构的结构化 CSV 数据集：

### 1. 历史表现（任务 1）
| 特征 | 类型 | 描述 |
| :--- | :--- | :--- |
| `prev_medals` | 整型 | 该国在前几届奥运周期中获得的奖牌总数。 |
| `prev_major_medals`| 整型 | 在历史优势项目上获得的奖牌数。 |
| `is_host` | 二进制 | 东道国状态标识 (1/0)。 |
| `next_events` | 整型 | 目标届次计划举行的项目数量。 |

### 2. 社会经济指标（任务 2）
| 特征 | 范围 | 描述 |
| :--- | :--- | :--- |
| `alpha` | {0, 1} | 国家政治稳定性的二进制指标。 |
| `beta` | {0, 0.5, 1} | 国家 GDP 水平的类别分类。 |
| `gamma` | 浮点型 | 项目奖牌分配程度（计算为 $1 - HHI$）。 |
| `delta` | 浮点型 | 竞争强度（计算为 $1 / n_{participants}$）。 |
| `epsilon`| {0, 1} | 运动员综合实力的二进制指标。 |

### 3. 赛事级记录（任务 3 & 4）
| 特征 | 类型 | 描述 |
| :--- | :--- | :--- |
| `Year` | 整型 | 特定奥运年份。 |
| `Country` | 字符串 | 参赛组织/国家名称。 |
| `Sport` | 字符串 | 具体体育项目（如田径、排球）。 |
| `Medal_Value` | 整型 | 量化得分（金牌=3，银牌=2，铜牌=1，无奖牌=0）。 |

## 验证与性能

该框架已针对 1992 年至 2024 年的历史数据进行了验证，观察结果如下：

* **预测精度**：堆叠集成模型在金牌预测中的均方误差 (MSE) 约为 **0.043**。
* **统计显著性**：东道国身份与奖牌总数之间的相关性得到了证实，$p$ 值为 **4.39E-278**。
* **执教影响**：分析表明，精英教练对目标国家队的贡献在统计上是显著的（$p < 0.05$）。
