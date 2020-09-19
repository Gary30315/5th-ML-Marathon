import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

pd.set_option('display.width',1000,'display.max_row',1000)
app_train = pd.read_csv('application_train.csv')
le = LabelEncoder()

for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder, 以加入相關係數檢查
            app_train[col] = le.fit_transform(app_train[col])            
# print(app_train.shape)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

# print(app_train.isnull().apply(sum))
# print(app_train['OWN_CAR_AGE'].head(100))
app_train['OWN_CAR_AGE'].dropna(inplace=True)
ages = app_train['OWN_CAR_AGE'].sort_values(ascending=False)
ages = app_train['OWN_CAR_AGE'].drop([271741,294131])
age_cut = pd.cut(ages,4)

# print(age_cut.value_counts())
new = pd.concat([ages,app_train['TARGET']],axis=1)
new = new.dropna()
# print(new.head())
new = new.groupby('OWN_CAR_AGE').mean()
new = new.reset_index()
print(new.head())
# px = new['OWN_CAR_AGE']
# py = new['TARGET']
# sns.barplot(px, py)
# plt.show()