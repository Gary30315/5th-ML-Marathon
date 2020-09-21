import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

app_train = pd.read_csv('application_train.csv')

unique_house_type = app_train['HOUSETYPE_MODE'].unique()

nrows = len(unique_house_type)
ncols = nrows // 2

plt.figure(figsize=(10,30))
for i in range(len(unique_house_type)):
    plt.subplot(nrows, ncols, i+1)

    app_train.loc[app_train['HOUSETYPE_MODE'] == unique_house_type[i],'AMT_CREDIT'].hist()
    
    plt.title(str(unique_house_type[i]))
plt.show()   