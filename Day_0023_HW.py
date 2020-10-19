import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data_path = 'Data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
# print(df.head())

num_features = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'float64' or dtype =='int64': 
        num_features.append(feature)

# print(f'{len(num_features)} Numeric Features : {num_features}\n')

df = df[num_features]
df = df.fillna(0)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]
# print(df.head())

# sns.distplot(df['Fare'][:train_num])
# plt.show()

df_mm = MMEncoder.fit_transform(df)
train_X = df_mm[:train_num]
estimator = LogisticRegression()
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#作業1
# df_fixed = copy.deepcopy(df)
# df_fixed['Fare'] = np.log1p(df_fixed['Fare'])

# sns.distplot(df_fixed['Fare'][:train_num])
# plt.show()

# df_fixed = MMEncoder.fit_transform(df_fixed)
# train_X = df_fixed[:train_num]
# estimator = LogisticRegression()
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

#作業2
df_fixed = copy.deepcopy(df)
df_num = df_fixed.shape[0]
print(df_num)
for i in range(df_num):
    if df_fixed['Fare'][i] <=0:
        df_fixed['Fare'][i] =1
# df_fixed['Fare'] = stats.boxcox(df_fixed['Fare'])[0]
# sns.distplot(df_fixed['Fare'][:train_num])
# plt.show()

# df_fixed = MMEncoder.fit_transform(df_fixed)
# train_X = df_fixed[:train_num]
# estimator = LogisticRegression()
# print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

