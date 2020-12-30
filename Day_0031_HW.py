import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

data_path = 'Data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df =df.drop(['PassengerId', 'Survived'],axis=1)

LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))

# print(df.head())

estimator = RandomForestRegressor()
estimator.fit(df.values, train_Y)

feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)

#作業1
#原特徵 準確率為0.444
# train_X = MMEncoder.fit_transform(df)
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#重要特徵 準確率為0.406
# high_feature = list(feats[:5].index)
# train_X = MMEncoder.fit_transform(df[high_feature])
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#作業2 加入4特徵準確率為0.4489
df['Add_char'] = (df['Sex'] + df['Name']) / 2
df['Multi_char'] = df['Sex'] * df['Name']
df['GO_div1p'] = df['Sex'] / (df['Name']+1) * 2
df['OG_div1p'] = df['Sex'] / (df['Name']+1) * 2
train_X = MMEncoder.fit_transform(df)
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
