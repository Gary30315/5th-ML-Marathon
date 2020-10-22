import pandas as pd 
import numpy as np 
import copy,time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

data_path = 'Data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'],axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])

# print(df.head())

object_features = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype =='object':
        object_features.append(feature)
# print(f'{len(object_features)} Numeric Features : {object_features}\n')

df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
# print(df.head())

df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')

df_temp1 = pd.get_dummies(df)
train_X = df_temp1[:train_num]
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')



#作業1
# 在房價預測中 One Hot Encoder 對線性回歸模型影響最大

#作業二
# One Hot Encode 對鐵達尼號生存資料預測準確分數提升不多 時間卻是Label Encoder的4倍