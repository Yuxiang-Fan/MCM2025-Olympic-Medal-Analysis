# Olympic Medal Trend Prediction: A Multi-Model Framework (MCM 2025)

This project implements a multi-stage mathematical framework to analyze and forecast Olympic medal distributions for the **2028 Los Angeles Summer Olympics**. Developed for the **2025 MCM/ICM Problem C**, the design utilizes a combination of ensemble learning, regularization, and causal inference to evaluate the impact of socio-economic factors and strategic coaching on national athletic performance.

## Implementation Characteristics

* **Ensemble Predictive Architecture**: A stacked regressor is implemented using **Random Forest** and **Gradient Boosting (GBDT)**. The model incorporates a secondary linear regression layer to fuse non-linear predictions and generate 95% confidence intervals through self-sampling .
* **Quantified Probability Scoring**: A multi-indicator scoring system is established to evaluate the "first-medal" potential of nations with no prior Olympic success. The model integrates political stability, GDP, and project-specific competition intensity (HHI) .
* **Lasso-Based Feature Selection**: Lasso regression is employed to identify the contribution coefficients of various sports disciplines. [cite_start]This module includes a **Variance Inflation Factor (VIF)** pre-check to mitigate multicollinearity between event settings .
* **Causal Inference of "Great Coach" Effect**: A **Difference-in-Differences (DID)** model is utilized to quantify the performance gain attributed to elite coaching (e.g., Lang Ping). [cite_start]The framework supports effect migration to target countries based on cosine similarity of historical performance .

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
| `gamma` | float | Event medal allocation degree (calculated as 1 - HHI). |
| `delta` | float | Competition intensity (calculated as 1 / n_participants). |
| `epsilon`| {0, 1} | Binary indicator for comprehensive athlete strength. |

### 3. Event-Level Records (Task 3 & 4)
| Feature | Type | Description |
| :--- | :--- | :--- |
| `Year` | int | The specific Olympic year. |
| `Country` | str | Name of the participating organization. |
| `Sport` | str | The specific discipline (e.g., Athletics, Volleyball). |
| `Medal_Value` | int | [cite_start]Quantified score (Gold=3, Silver=2, Bronze=1, None=0). |

## Validation and Performance

The framework has been verified against historical data from 1992 to 2024 with the following observations:

* [cite_start]**Predictive Accuracy**: Stacking ensemble models exhibit a mean squared error (MSE) of approximately **0.043** for gold medal forecasts .
* [cite_start]**Statistical Significance**: The correlation between host country status and total medal count was confirmed with a p-value of **4.39E-278** .
* [cite_start]**Coaching Impact**: Analysis indicates that elite coaching contributions are statistically significant (p < 0.05) for target national teams ].