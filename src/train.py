import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)


# Prepare for training
dv = DictVectorizer(sparse=False)

X_train_data = pd.read_csv(str(Config.DATASET_PATH / "train_data.csv"))
X_train_df = X_train_data.reset_index(drop=True)
X_train_dict = X_train_df.to_dict(orient='records')
X_train = dv.fit_transform(X_train_dict)

y_train_data = pd.read_csv(str(Config.DATASET_PATH / "train_target.csv"))
y_train = y_train_data.reset_index(drop=True)


# Build and fit the model

model = RandomForestRegressor(
    n_estimators=150, max_depth=6, random_state=Config.RANDOM_SEED
)
model = model.fit(X_train, y_train.to_numpy().ravel())

pickle.dump(model, open(str(Config.MODELS_PATH / "model.pickle"), "wb"))
