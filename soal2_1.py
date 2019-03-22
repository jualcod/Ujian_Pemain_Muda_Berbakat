import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================
# Establish Dataset, Pre-processing
# =============================================
# Importing
df = pd.read_csv('data.csv')
df = df[['Name', 'Age', 'Overall', 'Potential']]
df = df.fillna(np.NaN)
# print(df.head())
# print(df.info())	# Clean

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
df_targeted		= df[df.Targets == 1]
df_nontarget	= df[df.Targets == 0]

# =============================================
# Plotting
# =============================================
def plot_Age_Overall():
	plt.scatter(
		x=df_targeted.Age,
		y=df_targeted.Overall,
		s=5,
		label='Target',
		color='g'
	)
	plt.scatter(
		x=df_nontarget.Age,
		y=df_nontarget.Overall,
		s=5,
		label='Non-Target',
		color='r',
	)	
	plt.title('Age vs Overall')
	plt.xlabel('Age')
	plt.ylabel('Overall')
	plt.legend()
	plt.grid(True)

def plot_Age_Potential():
	plt.scatter(
		x=df_targeted.Age,
		y=df_targeted.Potential,
		s=5,
		label='Target',
		color='g'
	)
	plt.scatter(
		x=df_nontarget.Age,
		y=df_nontarget.Potential,
		s=5,
		label='Non-Target',
		color='r',
	)	
	plt.title('Age vs Potential')
	plt.xlabel('Age')
	plt.ylabel('Potential')
	plt.legend()
	plt.grid(True)

plt.figure('soal2_1')

plt.subplot(121)
plot_Age_Overall()

plt.subplot(122)
plot_Age_Potential()

plt.show()
plt.clf()