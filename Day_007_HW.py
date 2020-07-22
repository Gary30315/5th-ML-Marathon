import pandas as pd 
import numpy as np 
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 1000, 'display.max_rows', 1000)
df_train = pd.read_csv('titanic_train.csv')
df_test = pd.read_csv('titanic_test.csv')
# print(df_train.shape)

train_Y = df_train['Survived']
# print(train_y.head())
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])


dtype_df = df.dtypes.reset_index()
# print(dtype_df)
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
# print(dtype_df)

type_int = []
type_float=[]
type_object=[]

for dtype,feature in zip(df.dtypes,df.columns):
    if dtype == 'int64':
        type_int.append(feature)
    elif dtype == 'float64':
        type_float.append(feature)
    else:
        type_object.append(feature)
# print(type_int,type_float,type_object)

#作業1 
print(df[type_int].mean())
print(df[type_int].max())
print(df[type_int].nunique())

print(df[type_float].mean())
print(df[type_float].max())
print(df[type_float].nunique())

print(df[type_object].mean())
print(df[type_object].max())
print(df[type_object].nunique())

#在操作object類型時會出問題，這邊object為字串資料，沒辦法做數字型態資料的取平均/最大值/相異值等操作

#作業2
"""
boolean,二元的數字型資料類別。
object類別最難處理
"""