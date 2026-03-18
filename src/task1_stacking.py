import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')


def load_task1_data():
    """
    [Data Loading]
    Loads the historical Olympic dataset required for Task 1 model training and evaluation.

    Expected DataFrame Schema (N_samples, 10):
    - 'Country': Name of the participating organization (str)
    - 'prev_medals': Total medals won in previous Olympic Games (int)
    - 'prev_events': Establishment of events in previous Olympic Games (int)
    - 'prev_major_medals': Medals won in major events in previous Olympic Games (int)
    - 'is_host': Binary indicator if the country was the host (1/0)
    - 'next_events': Competition events in the next Olympic Games (int)
    - 'next_is_host': Binary indicator if the country is the host in the next Games (1/0)
    - 'Gold': Target variable for Gold medals (int)
    - 'Silver': Target variable for Silver medals (int)
    - 'Bronze': Target variable for Bronze medals (int)
    """
    # 实际数据加载逻辑
    df = pd.read_csv('data/task1_historical_features.csv')

    # 验证必要列是否存在
    required_columns = [
        'Country', 'prev_medals', 'prev_events', 'prev_major_medals',
        'is_host', 'next_events', 'next_is_host', 'Gold', 'Silver', 'Bronze'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset")

    return df


def build_stacking_model():
    """
    Constructs the Stacking Regressor architecture.
    Base models: Random Forest and Gradient Boosting.
    Meta-model: Linear Regression.
    """
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        cv=5
    )
    return stacking_model


def get_bootstrap_confidence_interval(model, X_train, y_train, X_test, n_iterations=100):
    """
    Generates 95% prediction intervals using self-sampling (Bootstrapping) methodology.
    """
    bootstrapped_preds = []

    for _ in range(n_iterations):
        X_resampled, y_resampled = resample(X_train, y_train)

        model.fit(X_resampled, y_resampled)
        preds = model.predict(X_test)
        bootstrapped_preds.append(preds)

    bootstrapped_preds = np.array(bootstrapped_preds)

    # Extract lower and upper percentiles
    lower_bound = np.percentile(bootstrapped_preds, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_preds, 97.5, axis=0)

    mean_preds = np.round(np.mean(bootstrapped_preds, axis=0))
    lower_bound = np.round(lower_bound)
    upper_bound = np.round(upper_bound)

    return mean_preds, lower_bound, upper_bound


def main():
    try:
        df = load_task1_data()
    except FileNotFoundError:
        print("Error: Data file 'data/task1_historical_features.csv' not found.")
        print("Please ensure the data file exists in the correct path.")
        return
    except ValueError as e:
        print(f"Data validation error: {e}")
        return

    features = ['prev_medals', 'prev_events', 'prev_major_medals',
                'is_host', 'next_events', 'next_is_host']
    X = df[features]

    targets = ['Gold', 'Silver', 'Bronze']
    results = {}

    for target in targets:
        print(f"Initializing model training and validation for target: {target}")
        y = df[target]

        # Test size parameter configured to 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = build_stacking_model()
        model.fit(X_train, y_train)

        y_pred = np.round(model.predict(X_test))
        mse = mean_squared_error(y_test, y_pred)
        print(f"Performance Metric - [{target}] MSE: {mse:.4f}")

        # Execute bootstrap resampling for confidence interval bounds
        mean_preds, lower_b, upper_b = get_bootstrap_confidence_interval(
            model, X_train, y_train, X_test, n_iterations=50
        )

        results[target] = {
            'Predictions': mean_preds,
            'Lower_Bound': lower_b,
            'Upper_Bound': upper_b
        }

    print("\nExecution terminated successfully.")


if __name__ == "__main__":
    main()