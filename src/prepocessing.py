
import pandas as pd
import numpy as np
from pathlib import Path

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
    features = df.groupby("timestamp").agg(
        vm_category_count=("vm_category", "nunique"),
        virtual_core_count=("virtual_core_bucket", "nunique"),
        memory_gb_count=("memory_gb_bucket", "nunique")
    ).reset_index()

    # Combine features and targets
    data = features.merge(
        targets,
        on="timestamp",
        how="inner"
    )

    # Ensure chronological order after merging
    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Separate features (X) and targets (y)
    x = data.drop(columns=[
        "active_vm_count",
        "aggregate_cpu_utilization"
    ])

    y = data[
        [
            "active_vm_count",
            "aggregate_cpu_utilization"
        ]
    ]

    # Chronological train/test split:
    x_train = x.iloc[:110]
    x_test = x.iloc[110:]

    y_train = y.iloc[:110]
    y_test = y.iloc[110:]

    # Convert dataFrames to NumPy arrays
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test
