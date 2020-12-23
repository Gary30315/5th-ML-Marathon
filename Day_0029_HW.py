import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
data_path = 'Data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df =df.drop(['PassengerId', 'Survived'],axis=1)
# print(df.head())

mean_df = df.groupby(['Pclass'])['Age'].mean().reset_index()
mode_df = df.groupby(['Pclass'])['Age'].apply(lambda x: x.mode()[0]).reset_index()
temp = pd.merge(mean_df, mode_df, how='left', on=['Pclass'])
df = pd.merge(df, temp, how='left', on=['Pclass'])

num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
# print(f'{len(num_features)} Numeric Features : {num_features}\n')

df = df[num_features]
df = df.fillna(-1)
df_minus = df.drop(['Age_x','Age_y'] , axis=1)

MMEncoder = MinMaxScaler()
train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

train_X1 = MMEncoder.fit_transform(df_minus)
estimator = LinearRegression()
print(cross_val_score(estimator, train_X1, train_Y, cv=5).mean())

#加入新特徵表現較好些