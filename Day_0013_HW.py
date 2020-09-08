import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app_train = pd.read_csv('application_train.csv')
#print(len(app_train.columns))

#作業1
cut_rule =[-0.01,0,0.99,2,2.99,5,100] 
app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'].values, cut_rule,include_lowest=True)
print(app_train['CNT_CHILDREN_GROUP'].value_counts())
print(app_train['CNT_CHILDREN_GROUP'].head())

#作業2-1
grp = app_train['CNT_CHILDREN_GROUP']
grouped_df = app_train.groupby(grp)['AMT_INCOME_TOTAL']
print(grouped_df.mean())

#作業2-2

plt_column = app_train.groupby(grp)['AMT_INCOME_TOTAL']
plt_by = grouped_df.mean()
app_train.boxplot(column=plt_column,by = plt_by, showfliers = False, figsize=(12,12))
plt.suptitle('boxplot')
plt.show()