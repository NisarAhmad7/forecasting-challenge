from prepocessing import preproce
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from pathlib import Path

#find the path of project to use from that 
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_PATH))

# the modle path
MODEL_PATH = PROJECT_PATH / "models" / "gradient_boosting.pkl"
DATA_PATH = PROJECT_PATH / "data" / "vm_cpu_readings.csv"
OUTPUT_PATH = PROJECT_PATH / "predictions.csv"


def predict():

    x_train, x_test, y_train, y_test, timestamps = preproce()
    # Model Load
    model = joblib.load(MODEL_PATH)
    pred = model.predict(x_test)
    assert len(timestamps) == len(pred), "Timestamp count does not match predictions"

    # add prediction to data frame to have one predctions.csv file 
    predictions = pd.DataFrame({
        "timestamp": timestamps,
        "active_vm_count" : pred[:, 0],
        "aggregate_cpu_utilization" : pred[: , 1]
    })
    # Save the file :
    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"predictions.csv created successfully at {OUTPUT_PATH}")


if __name__ == "__main__":
    predict()



    