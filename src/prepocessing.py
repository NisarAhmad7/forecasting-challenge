
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "vm_cpu_readings.csv"
DATA_PATH_2 = BASE_DIR / "data" / "vm_metadata.csv"

def preproce():
    # Read the datasets
    df = pd.read_csv(DATA_PATH)
    meta_data = pd.read_csv(DATA_PATH_2)

    # Select useful metadata columns
    metadata_selected = meta_data[
        [
            "vm_id",
            "vm_category",
            "virtual_core_bucket",
            "memory_gb_bucket"
        ]
    ]

    df = df.merge(
        metadata_selected,
        on="vm_id",
        how="left"
    )

    # Create target variables
    targets = df.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean")
    ).reset_index()

    # IMPORTANT: Sort targets chronologically
    targets = targets.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Feature Engineering: Count VMs in each metadata category
    # Sort data chronologically
    targets = targets.sort_values("timestamp").reset_index(drop=True)

    # Chronological train/test split:
    # Keep the final 24 observations separate from training
    train = targets.iloc[:110].copy()
    test = targets.iloc[110:].copy()

    if len(train) != 110 or len(test) != 24:
        raise ValueError(
            f"Expected 110 training and 24 holdout observations, "
            f"but got {len(train)} and {len(test)}."
        )

    # Create lag features using only past values
    # Build training features from the training period only
    target_cols = ["aggregate_cpu_utilization", "active_vm_count"]

    for col in target_cols:
        for lag in [1, 2, 3]:
            train[f"{col}_lag_{lag}"] = train[col].shift(lag)

    # Remove rows without enough historical data
    lag_columns = [
        f"{col}_lag_{lag}"
        for col in target_cols
        for lag in [1, 2, 3]
    ]

    train = train.dropna(subset=lag_columns).reset_index(drop=True)

    # Features and targets
    x_train = train[lag_columns]
    y_train = train[["active_vm_count", "aggregate_cpu_utilization"]]

    # Keep holdout targets separate for evaluation
    # Do not use actual holdout values to create lag features
    y_test = test[["active_vm_count", "aggregate_cpu_utilization"]]

    # Convert dataFrames to NumPy arrays
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Keep the timestamps for the final 24 observations
    timestamps_test = test["timestamp"].to_numpy()

    # Historical values available at the forecast origin
    history = targets.iloc[:110][
        ["active_vm_count", "aggregate_cpu_utilization"]
    ].reset_index(drop=True)

    return x_train, y_train, y_test, timestamps_test, history


def forecast_future(model, history, steps=24):
    # Store predictions for the future timestamps
    history = history.copy()
    predictions = []

    target_cols = [
        "active_vm_count",
        "aggregate_cpu_utilization"
    ]

    lag_columns = [
        f"{col}_lag_{lag}"
        for col in ["aggregate_cpu_utilization", "active_vm_count"]
        for lag in [1, 2, 3]
    ]

    for _ in range(steps):
        features = {}

        # Create lag features from the available history
        for col in target_cols:
            for lag in [1, 2, 3]:
                features[f"{col}_lag_{lag}"] = history[col].iloc[-lag]

        x_next = pd.DataFrame(
            [[features[col] for col in lag_columns]]
        )

        # Predict the next timestamp
        prediction = model.predict(x_next)[0]

        predictions.append(prediction)

        # Add predicted values, not actual test values
        history.loc[len(history)] = prediction

    return np.array(predictions)


if __name__ == "__main__":
    # Prepare the training and holdout data
    x_train, y_train, y_test, timestamps_test, history = preproce()

    # Train the forecasting model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(x_train, y_train)

    # Forecast the final 24 timestamps without using their actual targets
    predictions = forecast_future(
        model,
        history,
        steps=24
    )

    # Create the required submission file
    submission = pd.DataFrame({
        "timestamp": timestamps_test,
        "active_vm_count": predictions[:, 0],
        "aggregate_cpu_utilization": predictions[:, 1]
    })

    submission.to_csv(
        BASE_DIR / "predictions.csv",
        index=False
    )

    print(submission.head())
    print("Total predictions:", len(submission))
    print("Predictions saved to:", BASE_DIR / "predictions.csv")
