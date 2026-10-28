
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
from pathlib import Path
import sys


# find the path of project to use from that 
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_PATH))
from src.prepocessing import preproce,forecast_future #here we wrrite becuase path can find the path if it be above the it can not find the path and that get error!


project_path = Path.cwd().parent
sys.path.insert(0, str(project_path))

# make a path where we can save the model
MODEL_PATH = PROJECT_PATH / "models" / "gradient_boosting.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def train():

    # preproce the data 
    x_train, y_train, y_test, timestamps_test, history = preproce()
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # making the Model this model had the good performance
    model = MultiOutputRegressor(GradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.001,
    n_estimators=300,
    criterion='squared_error',
    max_depth=3,
    subsample=0.8,
    random_state=42
    ))

    model.fit(x_train, y_train)
    pred = forecast_future(model, history, steps=24)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(f"MAE ---> {mae}")
    print(f"RMSE ---> {rmse}")

    print("=============== All Detailse =================")
    # To shoe for both target the out put of the MAE and RMSE
    print("Active VM Count MAE:",
        mean_absolute_error(y_test[:, 0], pred[:, 0]))

    print("CPU Utilization MAE:",
        mean_absolute_error(y_test[:, 1], pred[:, 1]))

    print("Active VM Count RMSE:",
        np.sqrt(mean_squared_error(y_test[:, 0], pred[:, 0])))

    print("CPU Utilization RMSE:",
        np.sqrt(mean_squared_error(y_test[:, 1], pred[:, 1])))  

    # save the MOdel
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved in this path {MODEL_PATH}")

if __name__=="__main__":
    
    train()





    
    

