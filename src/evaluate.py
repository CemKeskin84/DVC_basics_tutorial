import json
import math
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
from config import Config

Config.METRICS_PATH.mkdir(parents=True, exist_ok=True)

X_test = pd.read_csv(str(Config.PREPARED_DATASET_PATH / "test_data.csv"))
y_test = pd.read_csv(str(Config.PREPARED_DATASET_PATH / "test_target.csv"))

model = pickle.load(open(str(Config.MODELS_PATH / "model.pickle"), "rb"))

r_squared = model.score(X_test, y_test)

y_pred = model.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))

with open(str(Config.METRICS_FILE_PATH), "w") as outfile:
    json.dump(dict(r_squared=r_squared, rmse=rmse), outfile)
