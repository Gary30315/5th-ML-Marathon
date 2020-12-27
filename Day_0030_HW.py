import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
warnings.filterwarnings('ignore')
data_path = 'Data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']

corr = df.corr()
sns.heatmap(corr)
# plt.show()

df =df.drop(['PassengerId', 'Survived'],axis=1)

num_feature = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'int64' or dtype =='float64':
        num_feature.append(feature)
# print(f'{len(num_feature)} Numeric Features : {num_feature}\n')

df = df[num_feature]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()


high_list = list(corr[(corr['Survived']>0.1) | (corr['Survived']<-0.1)].index)
high_list.pop(0)
# print(high_list)

#作業1

# 原始特徵 + 線性迴歸 0.116
# train_X = MMEncoder.fit_transform(df)
# estimator = LinearRegression()
# score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
# print(f'cross_val_score: {score}')

# 高相關性特徵 + 線性迴歸 0.1022
# train_X = MMEncoder.fit_transform(df[high_list])
# score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
# print(f'cross_val_score: {score}')

# 原始特徵 + 梯度提升樹 0.1825
# train_X = MMEncoder.fit_transform(df)
# estimator = GradientBoostingRegressor()
# score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
# print(f'cross_val_score: {score}')

# 高相關性特徵 + 梯度提升樹 0.1430
# train_X = MMEncoder.fit_transform(df[high_list])
# score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
# print(f'cross_val_score: {score}')

#作業2
L1_Reg = Lasso(alpha=0.001)
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
L1_Reg.fit(train_X, train_Y)
# print(L1_Reg.coef_)



from itertools import compress
L1_mask = list((L1_Reg.coef_>0.2) | (L1_Reg.coef_<-0.2))
L1_list = list(compress(list(df), list(L1_mask)))
# print(L1_list)

# L1_Embedding 特徵 + 線性迴歸 0.1140
# train_X = MMEncoder.fit_transform(df[L1_list])
# estimator = LinearRegression()
# score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
# print(f'cross_val_score: {score}')

# L1_Embedding 特徵 + 梯度提升樹 0.17107
# train_X = MMEncoder.fit_transform(df[L1_list])
# estimator = GradientBoostingRegressor()
# score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
# print(f'cross_val_score: {score}')

'''
使用高相關性抽取的特徵皆無法提升分數
'''