import pandas as pd 
import numpy as np 
import copy,time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

data_path = 'Data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
# print(df.head())
object_features = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'object':
        object_features.append(feature)
# print(f'{len(object_features)} Numeric Features : {object_features}\n')

df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]

#作業1

df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_X = df_temp[:train_num]
estimator = LogisticRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')


data = pd.concat([df[:train_num],train_Y],axis=1)
for c in df.columns:
    mean_df = data.groupby([c])['Survived'].mean().reset_index()
    mean_df.columns = [c, f'{c}_mean']
    data = pd.merge(data,mean_df, on = c , how = 'left')
    data = data.drop([c],axis=1)    
data = data.drop(['Survived'],axis=1)
value = cross_val_score(estimator, data, train_Y, cv=5).mean()
print(f'shape : {train_X.shape}')
print(f'score : {value}')
print(f'time : {time.time() - start} sec')

#作業2
df_temp = pd.DataFrame()
for c in df.columns:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
train_X = df_temp[:train_num]
estimator = LinearRegression()
start = time.time()
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')