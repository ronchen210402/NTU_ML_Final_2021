import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# %%

# 加權係數
weight = dict(BTC=6.78, ETH=5.89, BNB=4.31, ADA=4.40)

# %%

BTC = pd.read_csv('./new_Data/BTC.csv')[::-1]
ETH = pd.read_csv('./new_Data/ETH.csv')[::-1]
BNB = pd.read_csv('./new_Data/BNB.csv')[::-1]
ADA = pd.read_csv('./new_Data/ADA.csv')[::-1]

# %%

BTC = BTC.set_index(BTC['Date'])
ETH = ETH.set_index(ETH['Date'])
BNB = BNB.set_index(BNB['Date'])
ADA = ADA.set_index(ADA['Date'])

# %%

BTC_Close = BTC['Price']
ETH_Close = ETH['Price']
BNB_Close = BNB['Price']
ADA_Close = ADA['Price']

# %%

BTC_Train = BTC_Close.loc['1-Jan-18':'1-Jan-21'].values.reshape(-1, 1)
ETH_Train = ETH_Close.loc['1-Jan-18':'1-Jan-21'].values.reshape(-1, 1)
BNB_Train = BNB_Close.loc['1-Jan-18':'1-Jan-21'].values.reshape(-1, 1)
ADA_Train = ADA_Close.loc['1-Jan-18':'1-Jan-21'].values.reshape(-1, 1)

BTC_Test = BTC_Close.loc['1-Jan-21':].values.reshape(-1, 1)
ETH_Test = ETH_Close.loc['1-Jan-21':].values.reshape(-1, 1)
BNB_Test = BNB_Close.loc['1-Jan-21':].values.reshape(-1, 1)
ADA_Test = ADA_Close.loc['1-Jan-21':].values.reshape(-1, 1)

# %%

BTC_scaler = MinMaxScaler(feature_range=(0, 1))
BTC_std = BTC_scaler.fit_transform(BTC_Train)

ETH_scaler = MinMaxScaler(feature_range=(0, 1))
ETH_std = ETH_scaler.fit_transform(ETH_Train)

BNB_scaler = MinMaxScaler(feature_range=(0, 1))
BNB_std = BNB_scaler.fit_transform(BNB_Train)

ADA_scaler = MinMaxScaler(feature_range=(0, 1))
ADA_std = ADA_scaler.fit_transform(ADA_Train)

# %%

new_data = (weight['BTC'] * BTC_std + weight['ETH'] * ETH_std + weight['BNB'] * BNB_std + weight[
    'ADA'] * ADA_std) / np.sum(
    list(weight.values()))

# %%

time_stamp, output = 10, 1

point = int(new_data.shape[0] * 0.8)
train = new_data[:point]
valid = new_data[point:]

# train data
x_train, y_train = [], []
for i in range(time_stamp, len(train) - output):
    x_train.append(train[i - time_stamp:i])
    y_train.append(train[i: i + output])

x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, output)

# valid data
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid) - output):
    x_valid.append(valid[i - time_stamp:i])
    y_valid.append(valid[i: i + output])

x_valid, y_valid = np.array(x_valid), np.array(y_valid).reshape(-1, output)


# %%


def gen_test(scaler, test):
    scaled_data = scaler.fit_transform(test)
    x_test, y_test = [], []
    for i in range(time_stamp, len(test) - output):
        x_test.append(scaled_data[i - time_stamp:i])
        y_test.append(scaled_data[i: i + output])

    x_test, y_test = np.array(x_test), np.array(y_test).reshape(-1, output)

    return x_test, y_test


# %%

BTC_x_test, BTC_y_test = gen_test(BTC_scaler, BTC_Test)
ETH_x_test, ETH_y_test = gen_test(ETH_scaler, ETH_Test)
BNB_x_test, BNB_y_test = gen_test(BNB_scaler, BNB_Test)
ADA_x_test, ADA_y_test = gen_test(ADA_scaler, ADA_Test)

# %%

# create model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1:])))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.1))
model.add(Dense(output))

model.compile(optimizer=Adam(learning_rate=0.01), loss='mae', metrics=['acc'])

callBack = [ModelCheckpoint(f'./model/weight_model.h5',
                            monitor='val_loss', verbose=0, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-7)]

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=50,
                    validation_data=(x_valid, y_valid),
                    verbose=1,
                    callbacks=callBack)

# %%

plt.figure(figsize=(8, 6))
plt.title(f'weight_model')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.tight_layout()
plt.legend()
plt.savefig(f'weight_model.png')
plt.show()

# %%

time_stamp, output = 10, 1


def gen_train_valid(new_data):
    point = int(new_data.shape[0] * 0.8)
    train = new_data[:point]
    valid = new_data[point:]

    # train data
    x_train, y_train = [], []
    for i in range(time_stamp, len(train) - output):
        x_train.append(train[i - time_stamp:i])
        y_train.append(train[i: i + output])

    x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, output)

    # valid data
    x_valid, y_valid = [], []
    for i in range(time_stamp, len(valid) - output):
        x_valid.append(valid[i - time_stamp:i])
        y_valid.append(valid[i: i + output])

    x_valid, y_valid = np.array(x_valid), np.array(y_valid).reshape(-1, output)

    return x_train, y_train, x_valid, y_valid


# %%

BTC_x_train, BTC_y_train, BTC_x_valid, BTC_y_valid = gen_train_valid(BTC_std)

# %%

# create model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(BTC_x_train.shape[1:])))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.1))
model.add(Dense(output))

model.compile(optimizer=Adam(learning_rate=0.01), loss='mae', metrics=['acc'])

callBack = [ModelCheckpoint(f'./model/BTC_model.h5',
                            monitor='val_loss', verbose=0, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-7)]

history = model.fit(BTC_x_train, BTC_y_train,
                    batch_size=128,
                    epochs=50,
                    validation_data=(BTC_x_valid, BTC_y_valid),
                    verbose=1,
                    callbacks=callBack)

# %%

model = load_model('./model/weight_model.h5')


# %%

def predict(x_test, scaler):
    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred


# %%

BTC_y_pred_1 = predict(BTC_x_test, BTC_scaler)
ETH_y_pred_1 = predict(ETH_x_test, ETH_scaler)
BNB_y_pred_1 = predict(BNB_x_test, BNB_scaler)
ADA_y_pred_1 = predict(ADA_x_test, ADA_scaler)

# %%

model = load_model('./model/BTC_model.h5')

BTC_y_pred_2 = predict(BTC_x_test, BTC_scaler)
ETH_y_pred_2 = predict(ETH_x_test, ETH_scaler)
BNB_y_pred_2 = predict(BNB_x_test, BNB_scaler)
ADA_y_pred_2 = predict(ADA_x_test, ADA_scaler)

# %%

print('BTC:')
print('Weight model:')
print(f'MAE: {mean_absolute_error(BTC_y_test, BTC_y_pred_1)}')
print(f'RMSE: {np.sqrt(mean_squared_error(BTC_y_test, BTC_y_pred_1))}')
print()
print('BTC model:')
print(f'MAE: {mean_absolute_error(BTC_y_test, BTC_y_pred_2)}')
print(f'RMSE: {np.sqrt(mean_squared_error(BTC_y_test, BTC_y_pred_2))}')
print()

print('ETH:')
print('Weight model:')
print(f'MAE: {mean_absolute_error(ETH_y_test, ETH_y_pred_1)}')
print(f'RMSE: {np.sqrt(mean_squared_error(ETH_y_test, ETH_y_pred_1))}')
print()
print('BTC model:')
print(f'MAE: {mean_absolute_error(ETH_y_test, ETH_y_pred_2)}')
print(f'RMSE: {np.sqrt(mean_squared_error(ETH_y_test, ETH_y_pred_2))}')
print()

print('BNB:')
print('Weight model:')
print(f'MAE: {mean_absolute_error(BNB_y_test, BNB_y_pred_1)}')
print(f'RMSE: {np.sqrt(mean_squared_error(BNB_y_test, BNB_y_pred_1))}')
print()
print('BTC model:')
print(f'MAE: {mean_absolute_error(BNB_y_test, BNB_y_pred_2)}')
print(f'RMSE: {np.sqrt(mean_squared_error(BNB_y_test, BNB_y_pred_2))}')
print()

print('ADA:')
print('Weight model:')
print(f'MAE: {mean_absolute_error(ADA_y_test, ADA_y_pred_1)}')
print(f'RMSE: {np.sqrt(mean_squared_error(ADA_y_test, ADA_y_pred_1))}')
print()
print('BTC model:')
print(f'MAE: {mean_absolute_error(ADA_y_test, ADA_y_pred_2)}')
print(f'RMSE: {np.sqrt(mean_squared_error(ADA_y_test, ADA_y_pred_2))}')
print()
