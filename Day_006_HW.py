import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.width', 1000, 'display.max_rows', 1000)
dir_data = 'C:/Users/user/Documents/GitHub/5th-ML-Marathon'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
# print(sub_train.shape)
print(sub_train.head())
# print(app_train.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0))

# labelencoder = LabelEncoder()
# sub_train['WEEKDAY_APPR_PROCESS_START']=labelencoder.fit_transform(sub_train['WEEKDAY_APPR_PROCESS_START'])
# print(sub_train.head())

sub_train_one_hot_encoding = pd.get_dummies(sub_train)
print(sub_train_one_hot_encoding.head())
print(sub_train_one_hot_encoding.shape)
print(sub_train_one_hot_encoding.columns) 