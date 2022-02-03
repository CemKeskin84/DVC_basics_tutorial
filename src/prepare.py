import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config
import os

np.random.seed(Config.RANDOM_SEED)

df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH))
#df = pd.read_csv("./data/dataset.csv")


X = df.iloc[:,0:-2]
y = df.iloc[:,-2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=Config.RANDOM_SEED)

os.makedirs(os.path.join("data", "ready"), exist_ok=True)



#X_train.to_csv(str(Config.DATASET_PATH/ "train_data.csv"), index=None)
#X_test.to_csv(str(Config.DATASET_PATH / "test_data.csv"), index=None)
#y_train.to_csv(str(Config.DATASET_PATH/ "train_target.csv"), index=None)
#y_test.to_csv(str(Config.DATASET_PATH / "test_target.csv"), index=None)




X_train.to_csv(str(Config.DATASET_PATH/"ready"/ "train_data.csv"), index=None)
X_test.to_csv(str(Config.DATASET_PATH /"ready"/ "test_data.csv"), index=None)

y_train.to_csv(str(Config.DATASET_PATH/"ready"/ "train_target.csv"), index=None)
y_test.to_csv(str(Config.DATASET_PATH /"ready"/ "test_target.csv"), index=None)




