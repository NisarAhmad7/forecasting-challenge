
import pandas as pd
import numpy as np

def preproce():
    df = pd.read_csv("vm_cpu_readings.csv")

    targets= df.groupby("timestamp").agg(active_vm_count =('vm_id', 'unique'),
    aggrgate_cpu_utilization = ('avg_cpu', 'mean')).reset_index()


    x = targets[['timestamp']]
    y = targets[["active_vm_count","aggrgate_cpu_utilization"]]

    x_train = x.iloc[:110]
    x_test = x.iloc[110:]

    y_train = y.iloc[:110]
    y_test = y.iloc[110:]



    return x_train, x_test, y_train, y_test


