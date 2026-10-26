
import pandas as pd
import numpy as np
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "vm_cpu_readings.csv"

def preproce():
    df = pd.read_csv(DATA_PATH)

    targets= df.groupby("timestamp").agg(active_vm_count=('vm_id', 'nunique'),
    aggrgate_cpu_utilization = ('avg_cpu', 'mean')).reset_index()


    x = targets[['timestamp']]
    y = targets[["active_vm_count","aggrgate_cpu_utilization"]]

    #Split Data:
    x_train = x.iloc[:110]
    x_test = x.iloc[110:]

    y_train = y.iloc[:110]
    y_test = y.iloc[110:]
    
    #Change to Numpy
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test


