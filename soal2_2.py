import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Importing
df = pd.read_csv('data.csv')
df = df[['Name', 'Age', 'Overall', 'Potential']]
df = df.fillna(np.NaN)
# print(df.head())
print(df.info())	# Clean

# Processing
ind_Targets = []
bool_Targets = []

for n,x,y in zip(df.index, df.Age, df.Overall):
	if x <= 25 and y >= 80:
		ind_Targets.append(n)
		bool_Targets.append(1)
	else:
		bool_Targets.append(0)		
print(len(ind_Targets))
print(len(bool_Targets))

df['Targets'] = bool_Targets		# print(df.Targets.iloc[15])	# 1 True

# y Target Separation
df_targeted		= df[df.Targets == 1]			# df.species_name=='setosa' returns bool True/False
df_nontarget	= df[df.Targets == 0]
print(len(df_targeted))


# Initial Splitting
X = df[['Age', 'Overall', 'Potential']]
y = df.Targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# =====================================
# Machine Learning # Done
# =====================================
"""	"""
# Model Fitting

model	= LogisticRegression(multi_class='auto',solver='liblinear')
clf		= SVC()
randfor	= RandomForestClassifier()

model.fit(X_train, y_train)
clf.fit(X_train, y_train)
randfor.fit(X_train, y_train)

# =====================================
# Cross Validations # Done
# =====================================
"""	Cross Validation Score	"""

# CV Cross Validation Score
from sklearn.model_selection import cross_val_score

def CV_LogReg(X, y):
	return cross_val_score(
		LogisticRegression(),
		X=X,
		y=y,
		cv=5    # Default 3, will be changed to 5 in version 0.22.
	).round(4) * 100

def CV_SVC(X, y):
	return cross_val_score(
		SVC(gamma='auto'),
		X=X,
		y=y,
		cv=5    # Default 3, will be changed to 5 in version 0.22.
	).round(4) * 100

def CV_RF(X, y):
	return cross_val_score(
		RandomForestClassifier(n_estimators=10),
		X=X,
		y=y,
		cv=5    # Default 3, will be changed to 5 in version 0.22.
	).round(4) * 100


print('CV_LogReg\t:', CV_LogReg(X, y), 'mean:', CV_LogReg(X, y).mean().round(4), '%')
print('CV_SVC\t\t:', CV_SVC(X, y), 'mean:', CV_SVC(X, y).mean().round(4), '%')
print('CV_RandFor\t:', CV_RF(X, y), 'mean:', CV_RF(X, y).mean().round(4), '%')

# CV_LogReg       : [88.47 99.7  99.26 99.26 99.15] mean: 97.168 %
# CV_SVC          : [ 97.2   99.86 100.    99.97  99.89] mean: 99.384 %
# CV_RandFor      : [ 71.77 100.   100.    99.86  99.64] mean: 94.338 %