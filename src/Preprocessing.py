# %%
import pandas as pd

train = pd.read_csv('train.csv')

BNB = train[train['Asset_ID'] == 0]
BTC = train[train['Asset_ID'] == 1]
ADA = train[train['Asset_ID'] == 3]
ETH = train[train['Asset_ID'] == 6]

BNB.to_csv(f'ID_0_BNB.csv', index=None)
BTC.to_csv(f'ID_1_BTC.csv', index=None)
ADA.to_csv(f'ID_3_ADA.csv', index=None)
ETH.to_csv(f'ID_6_ETH.csv', index=None)

