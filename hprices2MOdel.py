import pandas as pd
import numpy as np
train_x = pd.read_csv("train_x-house.csv")
train_y = pd.read_csv("train_y-house.csv")
test_x = pd.read_csv("test_x-house.csv")
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
train_x["y"] = train_y
train_x = train_x[np.isfinite(train_x)]

from sklearn.linear_model import LinearRegression

#model = LinearRegression()

