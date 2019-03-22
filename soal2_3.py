import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# =============================================
# Establish Dataset, Pre-processing
# =============================================
# Importing Original Data
df = pd.read_csv('data.csv')
df = df[['Name', 'Age', 'Overall', 'Potential']]
df = df.fillna(np.NaN)
# print(df.head())
print(df.info())	# Clean

# =============================================
# Processing
# =============================================
ind_Targets = []
bool_Targets = []

for n,x,y in zip(df.index, df.Age, df.Overall):
	if x <= 25 and y >= 80:
		ind_Targets.append(n)
		bool_Targets.append(1)
	else:
		bool_Targets.append(0)		

df['Targets'] = bool_Targets		# print(df.Targets.iloc[15])	# 1 True

# y Target Separation
df_targeted		= df[df.Targets == 1]			# df.species_name=='setosa' returns bool True/False
df_nontarget	= df[df.Targets == 0]

# =====================================
# Machine Learning
# =====================================
"""	"""
# Splitting
X = df[['Age', 'Overall', 'Potential']]
y = df.Targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Model Fitting
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

model	= LogisticRegression(multi_class='auto',solver='liblinear')
model.fit(X_train, y_train)

# Importing Test Data
df_test = pd.read_csv('test_data.csv')
df_test = df_test.fillna(np.NaN)

X_new = df_test.iloc[:,1:]
df_test['Target_LogReg'] = model.predict(X_new)
df_test['Target_SVC'] = clf.predict(X_new)
print(df_test)

#                     Name  Age  Overall  Potential  Target_LogReg  Target_SVC
# 0       Andik Vermansyah   27       87         90              0           0
# 1     Awan Setho Raharjo   22       75         83              0           0
# 2      Bambang Pamungkas   38       85         75              0           0
# 3      Cristian Gonzales   43       90         85              0           0
# 4      Egy Maulana Vikri   18       88         90              1           0
# 5             Evan Dimas   24       85         87              1           1
# 6         Febri Hariyadi   23       77         80              0           0
# 7   Hansamu Yama Pranata   24       82         85              1           1
# 8  Septian David Maulana   22       83         80              1           0
# 9       Stefano Lilipaly   29       88         86              0           0