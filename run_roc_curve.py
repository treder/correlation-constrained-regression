'''
OLD VERSION: DOES NOT USE TEST SET (USES CROSS-VALIDATION ON TRAINING SET INSTEAD)
Performs regression for a range of rho values from 0 to 1.
Used to depict the ROC curve of rho vs MAE/ADC.
'''
import scipy, sys, time, pickle
import numpy as np
import pandas as pd
import sklearn, sklearn.datasets

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV

import correlation_constrained_regression as ccr
import analysis_tools as at

dataset = 'pac2019'
X, y = at.load_data(dataset)

tune_KernelRidge = {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}
tune_Ridge = {'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}

import time
n_iterations = 100
rho = np.linspace(0, 1, 51)

# iterations x correlation constraints x models (linear,ridge,kernelridge) x train/test phase
mse = np.zeros((n_iterations, len(rho), 3, 2))
corrs = np.zeros((n_iterations, len(rho), 3, 2))

print(f'Starting regression loop with {n_iterations} iterations')
for n in range(n_iterations):
    if n % 2 == 0: print('iteration', n)

    # get train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n)

    for m, bound in enumerate(rho): # constraints

        linreg = ccr.LinearRegression(correlation_bound=bound)
        ridge = GridSearchCV(ccr.Ridge(correlation_bound=bound), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
        kr = GridSearchCV(ccr.KernelRidge(correlation_bound=bound), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        # Fit models
        mse[n, m, 0, 0], mse[n, m, 0, 1], _, _, corrs[n, m, 0, 0], corrs[n, m, 0, 1] = at.fit_model(linreg, X_train, y_train, X_test, y_test)
        mse[n, m, 1, 0], mse[n, m, 1, 1], _, _, corrs[n, m, 1, 0], corrs[n, m, 1, 1] = at.fit_model(ridge, X_train, y_train, X_test, y_test)
        mse[n, m, 2, 0], mse[n, m, 2, 1], _, _, corrs[n, m, 2, 0], corrs[n, m, 2, 1] = at.fit_model(kr, X_train, y_train, X_test, y_test)

# save results
pickle.dump( (mse, corrs, rho, n_iterations), open(f'roc_curve_{dataset}.pickle', 'wb' ) )
