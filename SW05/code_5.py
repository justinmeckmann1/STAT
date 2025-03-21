#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:13:19 2024

@author: goedel
"""

# Aufgabe 5.3

import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('./data/Default.csv', sep=';')

# As a first inspection, print the first rows of the data:
print(df.head())
# As well as the dimensions of the set:
print('\nSize of Default =\n', df.shape)



# Add a numerical column for default and student
df = df.join(pd.get_dummies(df[['default', 'student']], 
                            prefix={'default': 'default', 
                                    'student': 'student'},
                            drop_first=True))

# Set ramdom seed
np.random.seed(1)
# Index of Yes:
i_yes = df.loc[df['default_Yes'] == 1, :].index

# Random set of No:
i_no = df.loc[df['default_Yes'] == 0, :].index
i_no = np.random.choice(i_no, replace=False, size=333)

i_ds = np.concatenate((i_no, i_yes))

# save downsampled dataframe:
df_ds = df.iloc[i_ds]

# Check dimensions:
print('\nSize of downsampled Default =\n', df_ds.shape)



import statsmodels.api as sm

y = df_ds['default_Yes']
x = df_ds['student_Yes']
x_sm = sm.add_constant(x)

model_stud = sm.GLM(y, x_sm, family=sm.families.Binomial())
model_stud = model_stud.fit()

print(model_stud.summary())



# Save regression coefficients
beta_0 = model_stud.params[0]
beta_1 = model_stud.params[1]

# Calculate probabilities:
prob_stud = 1 / (1 + np.exp( - (beta_0 + beta_1 * 1)))
prob_nonstud = 1 / (1 + np.exp( - (beta_0 + beta_1 * 0)))
print("probability on default given Student\n", 
      np.round(prob_stud, 4), 
      "\nprobability on default given not Student\n", 
      np.round(prob_nonstud, 4))

# Alternatively, directly predict using .predict():
prob_stud = model_stud.predict([1, 1])
prob_nonstud = model_stud.predict([1, 0])


import matplotlib.pyplot as plt

# split data based on student status
df_no = df.loc[df['student'] == 'No', :]
df_yes = df.loc[df['student'] == 'Yes', :]

# Create Figure and subplots
fig = plt.figure(figsize=(6, 5))
ax1 = fig.add_subplot(1, 1, 1)
ax1.boxplot([df_no['balance'], df_yes['balance']])
ax1.set_xlabel('Default')
ax1.set_ylabel('Balance')
ax1.set_xticklabels(['No','Yes'])

# plt.tight_layout()
plt.show()








# Aufgabe 5.4


import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('./data/auto.csv')

# As a first inspection, print the first rows of the data:
print(df.head())
# As well as the dimensions of the set:
print('\nSize of Auto =\n', df.shape)




# Create new variable
df['mpg01'] = np.zeros(df.shape[0], dtype=int)
for i in range(df.shape[0]):
    if df.loc[i, 'mpg'] > df['mpg'].median():
        df.loc[i, 'mpg01'] = int(1)
# Drop and add columns
df = df.drop(['mpg', 'name'], axis=1)

print(df.head())



""" Heatmap of correlations """
import seaborn as sns
import matplotlib.pyplot as plt

# Find correlations:
corr = df.drop(['origin', 'mpg01'], axis=1).corr()

fig = plt.figure(figsize = (10,8))
ax1 = fig.add_subplot(1, 1, 1)

sns.heatmap(corr)

plt.show()



""" Parallel coordinates using Pandas """
# Option 1, manualy scaling:
df_nor = df.drop(['origin'], axis=1).copy()
for col in df_nor.columns.values:
    df_nor[col] = df_nor[col] - df_nor[col].min()
    df_nor[col] = df_nor[col] / (df_nor[col].max() - df_nor[col].min())

# Option 2, scaling using sklearn:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_nor = df.drop(['origin'], axis=1).copy()

df_nor[df_nor.columns] = scaler.fit_transform(df_nor[df_nor.columns] )


# Plot parallel coordinates:
fig = plt.figure(figsize = (14,6))
ax = fig.add_subplot(1, 1, 1)
pd.plotting.parallel_coordinates(df_nor, 'mpg01', 
                                 ax=ax, color=('k', 'r'))
plt.show()



# Create matrix
confusion = pd.DataFrame({'mpg01': df['mpg01'],
                          'origin': df['origin']})
confusion = pd.crosstab(confusion.origin, confusion.mpg01,
                        margins=True, margins_name="Sum")

print(confusion)




# Redefine origin as a categorical variable
df = pd.get_dummies(data=df, drop_first=False, 
                    columns=['origin'])
df = df.rename(columns={'origin_1': 'American', 
                   'origin_2': 'European', 
                   'origin_3': 'Japanese'})

# Set ramdom seed
np.random.seed(2)

i = df.index
# Index of test
i_test = np.random.choice(i, replace=False, 
                          size=int(df.shape[0] / 5))

# Save DataFrames
df_test = df.iloc[i_test]
df_train = df.drop(i_test)

# Check dimensions:
print('\nSize of Train =\n', df_train.shape, 
      '\nSize of Test =\n', df_test.shape)


import statsmodels.api as sm

y = df_train['mpg01']
x = df_train.drop(['mpg01'], axis=1)
x_sm = sm.add_constant(x)

model = sm.GLM(y, x_sm, family=sm.families.Binomial())
model = model.fit()

print(model.summary())



# Predict for train and test
def class_err(x, y, model):
    """ Find classification error for given 
    x, y and fitted model """
    y_pred = model.predict(x)
    # Round to 0 or 1
    y_pred = y_pred.round()
    # Classification error
    e = abs(y - y_pred).mean()
    return e

y_test = df_test['mpg01']
x_test = df_test.drop(['mpg01'], axis=1)
x_sm_test = sm.add_constant(x_test)

e_train = class_err(x_sm, y, model)
e_test = class_err(x_sm_test, y_test, model) 

print('Train error:\n', np.round(e_train, 4),
      '\nTest error:\n', np.round(e_test, 4))


# Aufgabe 5.5

import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('./data/auto.csv')
# Create new variable
df['mpg01'] = np.zeros(df.shape[0], dtype=int)
for i in range(df.shape[0]):
    if df.loc[i, 'mpg'] > df['mpg'].median():
        df.loc[i, 'mpg01'] = int(1)
# Drop and add columns
df = df.drop(['mpg', 'name'], axis=1)
# Redefine origin as a categorical variable
df = pd.get_dummies(data=df, drop_first=False, 
                    columns=['origin'])
df = df.rename(columns={'origin_1': 'American', 
                   'origin_2': 'European', 
                   'origin_3': 'Japanese'})
# Set ramdom seed
np.random.seed(2)

# Index of test
i_test = np.random.choice(df.index, replace=False, 
                          size=int(df.shape[0] / 5))

# Save DataFrames
df_test = df.iloc[i_test]
df_train = df.drop(i_test)

# Check dimensions:
print('\nSize of Train =\n', df_train.shape, 
      '\nSize of Test =\n', df_test.shape)



import statsmodels.api as sm

y_train = df_train['mpg01']
y_test = df_test['mpg01']

x_train = df_train.drop(['mpg01'], axis=1)
x_test = df_test.drop(['mpg01'], axis=1)

x_sm_train = sm.add_constant(x_train)
x_sm_test = sm.add_constant(x_test)

# Create and fit logistic regression model
model = sm.GLM(y_train, x_sm_train, 
               family=sm.families.Binomial())
model = model.fit()

# Predict on train and testset
y_pred_train = model.predict(x_sm_train).round()
y_pred_test = model.predict(x_sm_test).round()

# Create confusion matrix
confusion_train = pd.DataFrame({'predicted': y_pred_train,
                                'true': y_train})
confusion_test = pd.DataFrame({'predicted': y_pred_test,
                                'true': y_test})
confusion_train = pd.crosstab(confusion_train.predicted, 
                              confusion_train.true, 
                              margins=True, margins_name="Sum")
confusion_test = pd.crosstab(confusion_test.predicted, 
                              confusion_test.true, 
                              margins=True, margins_name="Sum")

print(confusion_test, '\n\n',
      confusion_train)


# Accuracy : (tp + tn) / (tp + fp + fn + tn )
tp = confusion_test[1][1]
tn = confusion_test[0][0]
fp = confusion_test[1][0]
fn = confusion_test[0][1]
Accuracy = (tp + tn) / (tp + fp + fn + tn )
print(np.round(Accuracy, 4))

# Precision: tp / (tp + fp)
Precision = tp / (tp + fp)
print(np.round(Precision, 4))

# Recall: tp / (tp + fn)
Recall = tp / (tp + fn)
print(np.round(Recall, 4))



# F1-Score: 2 * precision * recall / (precision +  recall)
F1_Score = 2 * Precision * Recall / (Precision +  Recall)
print(np.round(F1_Score, 4))

def class_a(alpha, probability):
    classification = np.zeros(len(probability), dtype=int)
    for i in range(len(probability)):
        if probability.iloc[i] > alpha:
            classification[i] = 1
      
    return classification




import matplotlib.pyplot as plt

n = 100

alpha = np.linspace(0, 1, n)

# Create defintion returning both recall and fpr:
def ROC_data(x, y, model, alpha):
    """ Return Recall and False Posite Rate
    for a given x, y, model and threshold alpha """
    y_pred = class_a(alpha, model.predict(x))

    tp = (y_pred[y_pred == y] == 1).sum()
    tn = (y_pred[y_pred == y] == 0).sum()
    fp = (y_pred[y_pred != y] == 1).sum()
    fn = (y_pred[y_pred != y] == 0).sum()
    # Recall: tp / (tp + fn)
    Recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return fpr, Recall

fpr_train, Recall_train = np.zeros(n), np.zeros(n)
fpr_test, Recall_test = np.zeros(n), np.zeros(n)

for i in range(n):
    fpr_train[i], Recall_train[i] = (ROC_data(
        x_sm_train, y_train, model, alpha[i]))
    fpr_test[i], Recall_test[i] = (ROC_data(
        x_sm_test, y_test, model, alpha[i]))

""" Plot ROC curve """
fig = plt.figure(figsize = (7,6))
ax = fig.add_subplot(1, 1, 1)

plt.plot(fpr_train, Recall_train, label='train')
plt.plot(fpr_test, Recall_test, label='test')
plt.plot([0, 1], [0, 1], ':', label='random gues')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.legend()
plt.show()





# AUC by right riemann sum
AUC_train, AUC_test = 0, 0
for i in range(n-1):
    AUC_train += Recall_train[i] * (fpr_train[i] - fpr_train[i + 1])
    AUC_test += Recall_test[i] * (fpr_test[i] - fpr_test[i + 1])
    
print("AUC train:\n", np.round(AUC_train, 4),
      "\nAUC test:\n", np.round(AUC_test, 4))



# Best classifier:
dist_train, dist_test = [], []
for i in range(n):
    dist_train.append(fpr_train[i]**2 + (1 - Recall_train[i])**2)
    dist_test.append(fpr_test[i]**2 + (1 - Recall_test[i])**2)

alpha_train = alpha[np.argmin(dist_train)]
alpha_test = alpha[np.argmin(dist_test)]

print("Best alpha according to train data:\n", 
      np.round(alpha_train, 3),
      "\nBest alpha according to test data:\n", 
      np.round(alpha_test, 3))















