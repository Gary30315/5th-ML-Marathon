import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.width', 1000, 'display.max_rows', 1000)
app_train = pd.read_csv('application_train.csv')
# print(app_train['DAYS_EMPLOYED'].head())

#作業
# corr = app_train.corr()['TARGET']
# print(corr)
# corr_top = corr.sort_values()
# corr_tail = corr.sort_values(ascending = False)
# print(corr_top.head(15))
# print(corr_tail.head(15))

plt.plot(app_train['TARGET'],np.log10(app_train['DAYS_BIRTH']),'.')
plt.show()








