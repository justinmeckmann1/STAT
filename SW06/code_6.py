# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:08:11 2024

@author: resta
"""

import matplotlib.pyplot as plt
import numpy as np

x1 = [3, 2, 4, 1, 2, 4, 4]
x2 = [4, 2, 4, 4, 1, 3, 1]
y = ['red', 'red', 'red', 'red', 'blue', 'blue', 'blue']

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(x1, x2, c=y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()



from sklearn import svm

# Rewrite x in the right format
x = np.concatenate(([x1], [x2]), axis=0).T

# Fit SVM
cost = 10
clf = svm.SVC(kernel='linear', C=cost)
clf.fit(x, y)

# Find the Hyperplane
beta1, beta2 = clf.coef_[0][0],  clf.coef_[0][1]
beta0 = clf.intercept_[0]
x1_hyperplane = np.linspace(1, 4, 2)
x2_hyperplane = - beta1 / beta2 * x1_hyperplane - beta0 / beta2

# Create figure 
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

# Plot hyperplane
ax.plot(x1_hyperplane, x2_hyperplane, '-k')

# # Plot scatter data
ax.scatter(x1, x2, c=y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.title("Maximal margin Hyperplane")
plt.show()

print('Beta 0:', np.round(beta0, 2), 
      'Beta 1:', np.round(beta1, 2),
      'Beta 2:', np.round(beta2, 2))

# Find support vectors
x1_suppvec = clf.support_vectors_[:, 0]
x2_suppvec = clf.support_vectors_[:, 1]

# # Create figure 
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

# Plot support vectors
ax.scatter(x1_suppvec, x2_suppvec,
           s=100, linewidth=1, 
           facecolors='none', edgecolors='k')

# Plot hyperplane
ax.plot(x1_hyperplane, x2_hyperplane, 'k-')

# Plot scatter data
ax.scatter(x1, x2, c=y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.title("Support Vectors and Maximal margin Hyperplane")
plt.show()




# Calculate the offset, using for example (2, 2)
offset = 2 + (beta0 + beta1*2) / beta2
# find the margins
x2_upper_margin = x2_hyperplane + offset
x2_lower_margin = x2_hyperplane - offset

# Create figure 
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

# Plot margins 
ax.plot(x1_hyperplane, x2_upper_margin, 'k--')
ax.plot(x1_hyperplane, x2_lower_margin, 'k--')

# Plot support vectors
ax.scatter(x1_suppvec, x2_suppvec,
           s=100, linewidth=1, 
           facecolors='none', edgecolors='k')

# Plot hyperplane
ax.plot(x1_hyperplane, x2_hyperplane, 'k-')

# Plot scatter data
ax.scatter(x1, x2, c=y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.title("Support Vectors and Maximal margin Hyperplane")
plt.show()










import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create data
n = 500
# x1 and x2 from uniform distribution
x1 = np.random.uniform(size=n) - 0.5
x2 = np.random.uniform(size=n) - 0.5

# y depending on x1 and x2:
y = []
for i in range(n):
    if (-0.2 * x1[i] + x2[i]) > 0:
        y.append('red')
    else:
        y.append('blue')

# Plot
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(x1, x2, c=y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create data
n = 500
# x1 and x2 from uniform distribution
x1 = np.random.uniform(size=n) - 0.5
x2 = np.random.uniform(size=n) - 0.5

# y depending on x1 and x2:
y = []
for i in range(n):
    if (-0.2 * x1[i] + x2[i]) > 0:
        y.append('red')
    else:
        y.append('blue')

# Plot
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(x1, x2, c=y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()


from sklearn import svm

costs = np.linspace(1, 50, 20)
error = []

# Rewrite x in the right format
x = np.concatenate(([x1], [x2]), axis=0).T

# for each cost, fit and find the cross val error:
for c in costs:
    # Fit SVM
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(x, y)
    # predict scores
    y_pred = clf.predict(x)
    # find error
    error_i = n - (y_pred == y).sum()
    error_i = error_i / n
    error.append(error_i)

# plot
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

ax.plot(costs, error, '-b')
ax.set_xlabel('cost')
ax.set_ylabel('train error')
plt.show() 



from sklearn.model_selection import GridSearchCV
# Set parameters to be tuned. Other options can be added
tune_parameters = {'C': costs}

# Tune SVM
clf_tune = GridSearchCV(svm.SVC(kernel='linear'), 
                        tune_parameters, 
                        cv=10)
clf_tune.fit(x, y)

# Save Tune scores:
error_tune = 1 - clf_tune.cv_results_['mean_test_score']

# plot
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

ax.plot(costs, error,
        '-b', alpha=0.8, label='train error')
ax.plot(costs, error_tune, 
        '-k', alpha=0.8, label='Cross validation error')
ax.set_xlabel('cost')
ax.set_ylabel('error')
plt.legend()
plt.show() 



# Create test data:
np.random.seed(44)
# x1 and x2 from uniform distribution
x1_test = np.random.uniform(size=n) - 0.5
x2_test = np.random.uniform(size=n) - 0.5

# y depending on x1 and x2:
y_test = []
for i in range(n):
    if (-0.2 * x1_test[i] + x2_test[i]) > 0:
        y_test.append('red')
    else:
        y_test.append('blue')

# Fit optimal model:
clf_opt = svm.SVC(kernel='linear', C=clf_tune.best_params_['C'])
clf_opt.fit(x, y)

# Prediction
x_test = np.concatenate(([x1_test], [x2_test]), axis=0).T
y_pred = clf_opt.predict(x_test)

# error:
error = n - (y_pred == y_test).sum()
error = error / n

print("Prediction error on test data:\n", np.round(error, 3))



