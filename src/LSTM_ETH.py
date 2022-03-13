# importing required libraries.
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

# 讀取資料
ETH = pd.read_csv('ID_6_ETH.csv')


# %%

def main(time_stamp, output):
    ETH_Close = ETH['Close']

    # time_stamp, output = 30, 5

    point_1 = int(ETH.shape[0] * 0.7)
    point_2 = int(ETH.shape[0] * 0.85)

    # 70% train, 15% valid, 15% test
    train = ETH_Close[:point_1 + time_stamp].values.reshape(-1, 1)
    valid = ETH_Close[point_1 - time_stamp:point_2 + time_stamp].values.reshape(-1, 1)
    test = ETH_Close[point_2 - time_stamp:].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train)

    # train data
    x_train, y_train = [], []
    for i in range(time_stamp, len(train) - output):
        x_train.append(scaled_data[i - time_stamp:i])
        y_train.append(scaled_data[i: i + output])

    x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, output)

    # valid data
    scaled_data = scaler.fit_transform(valid)
    x_valid, y_valid = [], []
    for i in range(time_stamp, len(valid) - output):
        x_valid.append(scaled_data[i - time_stamp:i])
        y_valid.append(scaled_data[i: i + output])

    x_valid, y_valid = np.array(x_valid), np.array(y_valid).reshape(-1, output)

    # test data
    scaled_data = scaler.fit_transform(test)
    x_test, y_test = [], []
    for i in range(time_stamp, len(test) - output):
        x_test.append(scaled_data[i - time_stamp:i])
        y_test.append(scaled_data[i: i + output])

    x_test, y_test = np.array(x_test), np.array(y_test).reshape(-1, output)

    # create model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1:])))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dropout(0.1))
    model.add(Dense(output))

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mae', metrics=['acc'])

    callBack = [ModelCheckpoint(f'./model/ETH_LSTM(Input={time_stamp}, Output={output}).h5',
                                monitor='val_loss', verbose=0, save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-7)]

    start = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=1024,
                        epochs=50,
                        validation_data=(x_valid, y_valid),
                        verbose=1,
                        callbacks=callBack)
    end = time.time()

    print("ETH", time_stamp, output)
    print(f'{((end - start) / 20):.2f}')

    plt.figure(figsize=(8, 6))
    plt.title(f'ETH_LSTM(Input={time_stamp}, Output={output})')
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.tight_layout()
    plt.savefig(f'./figure/ETH_LSTM(Input={time_stamp}, Output={output})_loss.png')
    plt.show()

    model = load_model(f'./model/ETH_LSTM(Input={time_stamp}, Output={output}).h5')

    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)

    y_pred_avg = np.average(y_pred, axis=1)

    x = np.arange(test.shape[0] - time_stamp - output)

    plt.figure(figsize=(16, 10))
    plt.title(f'ETH_LSTM(Input={time_stamp}, Output={output})')
    plt.plot(x, test[time_stamp + output:], 'b', linewidth=3, label='History')
    plt.plot(x, y_pred_avg, 'r', linewidth=1, label='Predict')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'./figure/ETH_LSTM(Input={time_stamp}, Output={output}).png')
    plt.show()

    print(f'MAE: {mean_absolute_error(test[time_stamp + output:], y_pred_avg):.3f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(test[time_stamp + output:], y_pred_avg)):.3f}')
    print()


# %%

main(15, 1)
