
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
from pathlib import Path
import sys



PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_PATH))
from src.prepocessing import preproce

project_path = Path.cwd().parent
sys.path.insert(0, str(project_path))

MODEL_PATH = PROJECT_PATH / "models" / "gradient_boosting.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def train():
    # preproce the data 
    x_train, x_test, y_train, y_test = preproce()

    # My Cheaked model with good performance
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
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(f"MAE ---> {mae}")
    print(f"RMSE ---> {rmse}")


    if MODEL_PATH.exists():
        print("Model Exist!")

    else:
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved in this path {MODEL_PATH}")

if __name__=="__main__":
    
    train()




    
    

