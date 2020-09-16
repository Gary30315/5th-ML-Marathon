import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

app_train = pd.read_csv('application_train.csv')
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

age_data = app_train[['TARGET','DAYS_BIRTH']]
age_data['YEAR_BIRTH'] = app_train['DAYS_BIRTH']/365

age_seperation = np.linspace(age_data['YEAR_BIRTH'].min(),age_data['YEAR_BIRTH'].max(),11)
# print(age_seperation)
age_data['YEARS_BINNED'] = pd.cut(age_data['YEAR_BIRTH'], bins = age_seperation)
# print(age_data['YEARS_BINNED'].value_counts())
year_group_sorted = age_data['YEARS_BINNED'].sort_values(ascending=True)
# print(year_group_sorted)

# plt.figure(figsize=(8,6))
# for i in range(len(year_group_sorted)):
#     sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
#                               (age_data['TARGET'] == 0), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
    
#     sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
#                               (age_data['TARGET'] == 1), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
# plt.title('KDE with Age groups')
# plt.show()

age_data = age_data.drop(['DAYS_BIRTH'],axis=1)
age_group = age_data.groupby('YEARS_BINNED').mean()
# print(age_group)

px = age_group['YEAR_BIRTH']
py = age_group['TARGET']
sns.barplot(px, py)
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')
plt.show()