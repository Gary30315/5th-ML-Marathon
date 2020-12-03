import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import copy, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer #只考慮每種詞彙出現頻率
from sklearn.feature_extraction.text import TfidfVectorizer #除了頻率，還考慮詞彙在樣本總體中出現頻率倒數

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

data_path = 'Data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])

object_features = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype =='object':
        object_features.append(feature)

df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
# print(df.columns)


#作業1
#標籤編碼
# df_temp = pd.DataFrame()
# for c in df.columns:
#     df_temp[c] = LabelEncoder().fit_transform(df[c])
# train_X = df_temp[:train_num]
# estimator = LogisticRegression()
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

count_df = df.groupby(['Cabin']).count()['Name'].reset_index()
count_df = count_df.rename(columns={'Name':'Cabin_Count'})
df = pd.merge(df, count_df, on=['Cabin'], how='left')
count_df.sort_values(by=['Cabin_Count'], ascending=False)

#計數編碼
# df_temp = pd.DataFrame()
# for c in object_features:
#     df_temp[c] = LabelEncoder().fit_transform(df[c])
# df_temp['Cabin_Count'] = df['Cabin_Count']
# train_X = df_temp[:train_num]
# estimator = LogisticRegression()
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#特徵雜湊
# df_temp = pd.DataFrame()
# for c in object_features:
#     df_temp[c] = LabelEncoder().fit_transform(df[c])
# df_temp['Cabin_Hash'] = df['Cabin'].map(lambda x:hash(x)%10)
# df_temp=df_temp.drop(['Cabin'],axis=1)
# train_X = df_temp[:train_num]
# estimator = LogisticRegression()
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
# print(df_temp.head())

#作業2
# 計數>特徵>標籤








