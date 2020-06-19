import os
import numpy as np
import pandas as pd

pd.set_option('display.width', 1000, 'display.max_rows', 1000)
dir_data = 'C:/Users/user/Desktop/100day practice/data/'
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: {}' .format(f_app))

app_train = pd.read_csv(f_app)
print(app_train.head(10)) #查看前10rows
print(app_train.tail(10)) #查看後10rows
print(app_train.iloc[:5, :5])#查看前5rows 前5columns
print(app_train.head(10).T)#前10筆資料drow跟column轉置
print(app_train.head(10).dropna(axis=0))#刪除有缺失值的row
print(app_train.head(10).fillna(value=0))#填補缺失值為0
print(app_train.head(10).isnull().sum())#查詢各feature缺失值數量
print(app_train.info())#查詢資料資訊




