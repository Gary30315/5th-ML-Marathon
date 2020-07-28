#作業1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000, 'display.max_rows', 1000)
df=pd.read_csv('application_train.csv')
# print(df.describe())
# print(df.dtypes)

dtype_select = ['int64'and'float']
numeric_columns = df.columns[df.dtypes.isin(dtype_select)]
numeric_columns = list(df[numeric_columns].columns[list(df[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
# print(numeric_columns)

# for col in numeric_columns:
#     plt.xlabel(col)
#     plt.boxplot(df[col])
#     plt.show()

# a = df['OWN_CAR_AGE'].dropna()
# plt.hist(a)
# plt.show()


'''
(AMT_INCOME_TOTAL)

# print(df['AMT_INCOME_TOTAL'].describe()) 
# print(df.AMT_INCOME_TOTAL.value_counts())
cdf = df.AMT_INCOME_TOTAL.value_counts().sort_index().cumsum() #索引排序後累加
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) 
plt.ylim([-0.05,1.05]) 
plt.show()

plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) 
plt.show()

'''

'''
(REGION_POPULATION_RELATIVE)
print(df['REGION_POPULATION_RELATIVE'].describe())
cdf = df.REGION_POPULATION_RELATIVE.value_counts().sort_index().cumsum()
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.ylim([-0.05,1.05]) 
plt.show()

df['REGION_POPULATION_RELATIVE'].hist()
plt.show()

df['REGION_POPULATION_RELATIVE'].value_counts()

'''


# (OBS_60_CNT_SOCIAL_CIRCLE)
print(df['OBS_60_CNT_SOCIAL_CIRCLE'].describe())
cdf = df.OBS_60_CNT_SOCIAL_CIRCLE.value_counts().sort_index().cumsum()
plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min() * 0.95, cdf.index.max() * 1.05])
plt.ylim([-0.05,1.05]) 
plt.show()

df['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()

df['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False)
