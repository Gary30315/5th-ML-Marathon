import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

path = 'C:/Users/user/Documents/GitHub/5th-ML-Marathon/Data/'
df_train = pd.read_csv( path+'titanic_train.csv')
df_test =  pd.read_csv( path+'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
# print(f'{len(num_features)} Numeric Features : {num_features}\n')

df = df[num_features]
train_num = train_Y.shape[0]

#作業1
df_m1 = df.fillna(-1) 
train_X = df_m1[:train_num]
estimator = LogisticRegression()
v1 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(v1)

df_m2 = df.fillna(value=0)
train_X = df_m2[:train_num]
v2 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(v2)

df_m3 = df.fillna(method='ffill')
train_X = df_m3[:train_num]
v3 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(v3)

#作業2
df_temp1 = StandardScaler().fit_transform(df_m2)
train_X = df_temp1[:train_num]
c1 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(c1)

df_temp1 = MinMaxScaler().fit_transform(df_m2)
train_X = df_temp1[:train_num]
c2 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(c2)
