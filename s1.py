ltipledata.py


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

df = pd.read_csv(
    "ML/london_merged.csv",
    parse_dates=["timestamp"],
    index_col="timestamp"
)




print(df.shape)

df["hour"] = df.index.hour
df["day_of_month"] = df.index.day
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size: len(df)]
print(len(train), len(test))
"""
f_columns = ["t1", "t2", "hum", "wind_speed"]


f_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:, f_columns] = f_transformer.transform(
    train[f_columns].to_numpy()
)



cnt_transformer = RobustScaler()

cnt_transformer = cnt_transformer.fit(train[["cnt"]])

train["cnt"] = cnt_transformer.transform(train[['cnt']])
test["cnt"] = cnt_transformer.transform(test[["cnt"]])

"""

def create_dataset(X, y, time_steps=1):
    Xs, ys = [],[]
    for i in range(len(X)- time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i+ time_steps])
    return np.array(Xs), np.array(ys)


time_steps= 10

X_train, y_train = create_dataset(train, train.cnt, time_steps)
X_test, y_test = create_dataset(test, test.cnt, time_steps)

print(X_train.shape, y_train.shape)


model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss="mean_squared_error", optimizer="adam")


history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


2024-04-06 20:03:33.592390: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
(17414, 9)
15672 1742
(15662, 10, 13) (15662,)
Epoch 1/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 11s 16ms/step - loss: 2080340.6250 - val_loss: 3155923.0000
Epoch 2/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 1880403.7500 - val_loss: 2929727.2500
Epoch 3/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 1721388.6250 - val_loss: 2728351.0000
Epoch 4/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 1580580.8750 - val_loss: 2545737.2500
Epoch 5/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 1454246.5000 - val_loss: 2376370.5000
Epoch 6/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - loss: 1339937.1250 - val_loss: 2220948.5000
Epoch 7/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 11s 25ms/step - loss: 1238088.5000 - val_loss: 2079599.1250
Epoch 8/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 10s 23ms/step - loss: 1146696.6250 - val_loss: 1949873.1250
Epoch 9/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 14ms/step - loss: 1064352.8750 - val_loss: 1830131.0000
Epoch 10/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 14ms/step - loss: 988852.6875 - val_loss: 1719311.0000
Epoch 11/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 14ms/step - loss: 921107.1250 - val_loss: 1616908.3750
Epoch 12/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 859774.3750 - val_loss: 1519629.2500
Epoch 13/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 8s 19ms/step - loss: 808587.8125 - val_loss: 1452054.2500
Epoch 14/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - loss: 753347.4375 - val_loss: 1350570.7500
Epoch 15/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 14s 32ms/step - loss: 693657.1875 - val_loss: 1262651.8750
Epoch 16/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 8s 17ms/step - loss: 640803.8750 - val_loss: 1176551.0000
Epoch 17/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 12s 22ms/step - loss: 592779.8750 - val_loss: 1105287.5000
Epoch 18/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 14ms/step - loss: 552603.2500 - val_loss: 1035857.9375
Epoch 19/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 14ms/step - loss: 516372.2188 - val_loss: 972441.0625
Epoch 20/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 481552.9688 - val_loss: 918839.5000
Epoch 21/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 10s 13ms/step - loss: 450101.0625 - val_loss: 876709.9375
Epoch 22/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 14ms/step - loss: 418523.1250 - val_loss: 808862.3125
Epoch 23/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 393357.9375 - val_loss: 768758.3125
Epoch 24/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - loss: 367421.6875 - val_loss: 721877.1250
Epoch 25/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 9s 20ms/step - loss: 344192.1875 - val_loss: 683095.0625
Epoch 26/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - loss: 324309.9375 - val_loss: 650538.4375
Epoch 27/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - loss: 305331.3125 - val_loss: 617016.8750
Epoch 28/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 12s 20ms/step - loss: 284005.5938 - val_loss: 584151.3750
Epoch 29/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - loss: 268940.5312 - val_loss: 545715.1250
Epoch 30/30
441/441 ━━━━━━━━━━━━━━━━━━━━ 7s 17ms/step - loss: 253708.3750 - val_loss: 522654.5000
