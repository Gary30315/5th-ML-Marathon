import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

app_train = pd.read_csv('application_train.csv')
ages = pd.DataFrame({'age':[18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})

bins = [0,10,20,30,50,100]
ages["equal_with_age"] = pd.cut(ages['age'],bins=bins,include_lowest=False)
print(ages["equal_with_age"].value_counts())


