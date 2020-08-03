import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mode
df = pd.read_csv('application_train.csv')

#作業1
num = [np.linspace(0,99,100)]
print(five_num)
quantile_100 = [np.percentile(df[~df['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in num]
print(quantile_100)

#作業2.1
print(sum(df['AMT_ANNUITY'].isnull()))
q_50=df.loc[df['AMT_ANNUITY'].isnull(),'AMT_ANNUITY']
q_50=q_50.fillna(np.percentile(df[~df['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = 50))
print(q_50)

#作業2.2
value = df['AMT_CREDIT'].values
normalize_value = ((((value - min(value)) / (max(value) - (min(value))))-0.5) *2)
plt.hist(normalize_value)
plt.show()

#作業3
mode_goods_price = list(df['AMT_GOODS_PRICE'].value_counts().index)
print(mode_goods_price)
df.loc[df['AMT_GOODS_PRICE'].isnull(), 'AMT_GOODS_PRICE'] = mode_goods_price[0]
print((df['AMT_GOODS_PRICE'].isnull().agg(sum)))