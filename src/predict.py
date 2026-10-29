
import sys
from pathlib import Path
import pandas as pd
import joblib

# Find the project root first
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_PATH))

# Import after adding the project root
from src.prepocessing import preproce, forecast_future

# Model and output paths
MODEL_PATH = PROJECT_PATH / "models" / "gradient_boosting.pkl"
OUTPUT_PATH = PROJECT_PATH / "predictions.csv"


def predict():
    x_train, y_train, y_test, timestamps_test, history = preproce()

    # load the trained model
    model = joblib.load(MODEL_PATH)

    # generate recursive forecasts for the holdout timestamps
    pred = forecast_future(model, history, steps=len(timestamps_test))

    # Check that the prediction count matches the timestamps
    assert len(timestamps_test) == len(pred), (
        "Timestamp count does not match predictions"
    )

    # Create the predictions DataFrame
    predictions = pd.DataFrame({
        "timestamp": timestamps_test,
        "active_vm_count": pred[:, 0],
        "aggregate_cpu_utilization": pred[:, 1]
    })

    # Save predictions
    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"predictions.csv created successfully at {OUTPUT_PATH}")


if __name__ == "__main__":
    predict()
