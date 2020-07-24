import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

train_data = pd.read_csv('application_train.csv')
train_show=train_data['AMT_GOODS_PRICE'].agg(['mean','std'])
print(train_show)

plt.figure()
plt.hist(train_data['AMT_CREDIT'],bins=50,facecolor="steelblue")
plt.show()


