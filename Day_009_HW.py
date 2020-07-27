#作業1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv('application_train.csv')
# print(df.describe())
# print(df.dtypes)
dtype_select = ['int64'and'float']
numeric_columns = df.columns[df.dtypes.isin(dtype_select)]
numeric_columns = list(df[numeric_columns].columns[list(df[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print(numeric_columns)

for col in numeric_columns:
    plt.xlabel(col)
    plt.boxplot(df[col])
    plt.show()

# print(df['AMT_INCOME_TOTAL'].describe())

# def ecdf(data):
#     n = len(data)
#     x = np.sort(data)
#     y = np.arange(1, n + 1) / n
#     return x, y

# # print(ecdf(df['AMT_INCOME_TOTAL']))
# x_vers, y_vers = ecdf(df['AMT_INCOME_TOTAL'])
# plt.plot(x_vers, y_vers)
# plt.xlabel('AMT_INCOME_TOTAL')
# plt.ylabel('ECDF')
# plt.show()


# cdf = df['REGION_POPULATION_RELATIVE']
# plt.plot(list(cdf.index), cdf/cdf.max())
# plt.xlabel('Value')
# plt.ylabel('ECDF')
# plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
# plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

# plt.show()

# # 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
# plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
# plt.xlabel('Value (log-scale)')
# plt.ylabel('ECDF')

# plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

# plt.show()