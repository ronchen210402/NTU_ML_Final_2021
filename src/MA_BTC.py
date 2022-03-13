# %%
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %%

# 讀取資料(每個timestamp間隔為1分鐘)
BTC = pd.read_csv('ID_1_BTC.csv')

# %%

BTC_7Day = np.zeros(BTC.shape[0])
step = 60 * 24 * 7
for i in range(step, BTC.shape[0]):
    BTC_7Day[i] = np.average((BTC['Close'][i - step:i]))

BTC_21Day = np.zeros(BTC.shape[0])
step = 60 * 24 * 21
for i in range(step, BTC.shape[0]):
    BTC_21Day[i] = np.average((BTC['Close'][i - step:i]))

BTC_56Day = np.zeros(BTC.shape[0])
step = 60 * 24 * 56
for i in range(step, BTC.shape[0]):
    BTC_56Day[i] = np.average((BTC['Close'][i - step:i]))

BTC_200Day = np.zeros(BTC.shape[0])
step = 60 * 24 * 147
for i in range(step, BTC.shape[0]):
    BTC_200Day[i] = np.average((BTC['Close'][i - step:i]))

# %%

X = np.arange(BTC.shape[0])
timestamp = list(BTC['timestamp'][::43200])
timestamp = [str(datetime.fromtimestamp(i))[0:10] for i in timestamp]


# %%
plt.figure(figsize=(16, 10))

plt.plot(X, BTC['Close'], 'b', linewidth=3, label='History')
plt.plot(X[60 * 24 * 7:], BTC_7Day[60 * 24 * 7:], 'y', linewidth=2.5, label='7 Day Moving Average')
plt.plot(X[60 * 24 * 21:], BTC_21Day[60 * 24 * 21:], 'c', linewidth=2.5, label='21 Day Moving Average')
plt.plot(X[60 * 24 * 56:], BTC_56Day[60 * 24 * 56:], 'g', linewidth=2.5, label='56 Day Moving Average')
plt.plot(X[60 * 24 * 147:], BTC_200Day[60 * 24 * 147:], 'r', linewidth=2.5, label='147 Day Moving Average')

plt.legend(loc='upper left', fontsize=16)

plt.xticks(X[::43200], timestamp, rotation=90, fontsize=12)
plt.yticks(fontsize=24)

plt.title('BTC Moving Average', fontsize=24)
plt.tight_layout()
plt.savefig('./figure/BTC_MA.png')
plt.show()

# %%
