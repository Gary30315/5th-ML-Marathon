import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import seaborn as sns

df_train = pd.read_csv('train.csv.')
# print(df.head())

train_Y = np.log1p(df_train['SalePrice'])
# print(train_Y.head())
df = df_train.drop(['Id', 'SalePrice'] ,axis=1)

num_features = []
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'int64' or dtype == 'float64':
        num_features.append(feature)

df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]

sns.regplot(x = df['1stFlrSF'][:train_num], y=train_Y)
plt.show()


#作業1
df['1stFlrSF'] = df['1stFlrSF'].clip(800,2500)
sns.regplot(x = df['1stFlrSF'], y=train_Y)
plt.show()

train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(cross_val_score)

#作業2
keep_indexs = (df['1stFlrSF']> 800) & (df['1stFlrSF']< 2500)
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]
sns.regplot(x = df['1stFlrSF'], y=train_Y)
plt.show()

train_X = MMEncoder.fit_transform(df)
estimator = LinearRegression()
cross_val_score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(cross_val_score)











