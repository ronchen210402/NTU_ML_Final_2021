# %%
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %%

# 讀取資料(每個timestamp間隔為1分鐘)
ETH = pd.read_csv('ID_6_ETH.csv')

# %%

ETH_7Day = np.zeros(ETH.shape[0])
step = 60 * 24 * 7
for i in range(step, ETH.shape[0]):
    ETH_7Day[i] = np.average((ETH['Close'][i - step:i]))

ETH_21Day = np.zeros(ETH.shape[0])
step = 60 * 24 * 21
for i in range(step, ETH.shape[0]):
    ETH_21Day[i] = np.average((ETH['Close'][i - step:i]))

ETH_56Day = np.zeros(ETH.shape[0])
step = 60 * 24 * 56
for i in range(step, ETH.shape[0]):
    ETH_56Day[i] = np.average((ETH['Close'][i - step:i]))

ETH_200Day = np.zeros(ETH.shape[0])
step = 60 * 24 * 147
for i in range(step, ETH.shape[0]):
    ETH_200Day[i] = np.average((ETH['Close'][i - step:i]))

# %%

X = np.arange(ETH.shape[0])
timestamp = list(ETH['timestamp'][::43200])
timestamp = [str(datetime.fromtimestamp(i))[0:10] for i in timestamp]


# %%
plt.figure(figsize=(16, 10))

plt.plot(X, ETH['Close'], 'b', linewidth=3, label='History')
plt.plot(X[60 * 24 * 7:], ETH_7Day[60 * 24 * 7:], 'y', linewidth=2.5, label='7 Day Moving Average')
plt.plot(X[60 * 24 * 21:], ETH_21Day[60 * 24 * 21:], 'c', linewidth=2.5, label='21 Day Moving Average')
plt.plot(X[60 * 24 * 56:], ETH_56Day[60 * 24 * 56:], 'g', linewidth=2.5, label='56 Day Moving Average')
plt.plot(X[60 * 24 * 147:], ETH_200Day[60 * 24 * 147:], 'r', linewidth=2.5, label='147 Day Moving Average')

plt.legend(loc='upper left', fontsize=16)

plt.xticks(X[::43200], timestamp, rotation=90, fontsize=12)
plt.yticks(fontsize=24)

plt.title('ETH Moving Average', fontsize=24)
plt.tight_layout()
plt.savefig('./figure/ETH_MA.png')
plt.show()

# %%
