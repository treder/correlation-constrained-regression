'''
Performs regression for a range of rho values from 0 to 1
using the PAC2019 train and test sets.
Used to depict the ROC curve of rho vs MAE/ADC.
Bootstrapping is used to estimate SD of the estimates.
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
X_train, X_test, y_train, y_test, feature_names = at.load_data(dataset)
n_train, n_test = X_train.shape[0], X_test.shape[0]

tune_KernelRidge = {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}
tune_Ridge = {'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}

import time
n_bootstrap_iterations = 100
rho = np.linspace(0, 1, 51)

# iterations x correlation constraints x models (linear,ridge,kernelridge) x train/test phase
mae = np.zeros((n_bootstrap_iterations, len(rho), 3, 2))
corrs = np.zeros((n_bootstrap_iterations, len(rho), 3, 2))

print(f'Starting regression loop with {n_bootstrap_iterations} iterations')
for n in range(n_bootstrap_iterations):
    if n % 2 == 0: print('iteration', n)

    # bootstrap training data
    boot_ix = np.random.choice(np.arange(n_train), size=n_train, replace=True)
    X_boot, y_boot = X_train[boot_ix, :], y_train[boot_ix]

    for m, bound in enumerate(rho): # constraints

        linreg = ccr.LinearRegression(correlation_bound=bound)
        ridge = GridSearchCV(ccr.Ridge(correlation_bound=bound), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
        kr = GridSearchCV(ccr.KernelRidge(correlation_bound=bound), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        # Fit models
        mae[n, m, 0, 0], mae[n, m, 0, 1], _, _, corrs[n, m, 0, 0], corrs[n, m, 0, 1] = at.fit_model(linreg, X_boot, y_boot, X_test, y_test)
        mae[n, m, 1, 0], mae[n, m, 1, 1], _, _, corrs[n, m, 1, 0], corrs[n, m, 1, 1] = at.fit_model(ridge, X_boot, y_boot, X_test, y_test)
        mae[n, m, 2, 0], mae[n, m, 2, 1], _, _, corrs[n, m, 2, 0], corrs[n, m, 2, 1] = at.fit_model(kr, X_boot, y_boot, X_test, y_test)

# save results
pickle.dump( (mae, corrs, rho), open(f'roc_curve_{dataset}_train_test.pickle', 'wb' ) )
