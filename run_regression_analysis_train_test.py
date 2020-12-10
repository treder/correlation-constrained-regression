'''
Evaluates the three regression models (OLS, Ridge, Kernel Ridge)
on the official PAC2019 train and test sets.
Bootstrapping is used to estimate SD of the estimates.
'''
import scipy, sys, pickle
import numpy as np
import pandas as pd
import sklearn, sklearn.datasets

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.api as sm

sys.path.append('/home/matthias/mt03/python_tools')
import file_tools as ft
import correlation_constrained_regression as ccr
import analysis_tools as at

dataset = 'pac2019'
X_train, X_test, y_train, y_test, feature_names = at.load_data(dataset)
n_train, n_test = X_train.shape[0], X_test.shape[0]

tune_KernelRidge = [
  {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}
#,
#  {'kernel': ['poly'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10], 'degree': [2, 3, 4, 5], 'coef0':[0, 1]}
 ]
# tune_KernelRidge = {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}
tune_Ridge = {'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}

import time
n_bootstrap_iterations = 100
n_constraints = 7

# iterations x constraints (unconstrained,zero,bounded 0.1, 0.2, 0.3) x models (linear,ridge,kernelridge) x train/inference phase
times = np.zeros((n_bootstrap_iterations, n_constraints, 3, 2))
mae = np.zeros((n_bootstrap_iterations, n_constraints, 3, 2))
corrs = np.zeros((n_bootstrap_iterations, n_constraints, 3, 2))

def fit_model_and_correct(model, X_train, y_train, X_test, y_test, approach):
    '''
    Fits model, calculates MAE on train and test data and performs timing.
    '''
    # TRAIN
    # Step 1: Standard regression
    start_time = time.time()
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)

    # Step 2: Train correction coefficients on train data
    if approach == 1:
        delta_train = yhat_train - y_train
        # regress delta on y
        fit = sm.OLS(delta_train, sm.add_constant(y_train)).fit()
        b0, b1 = fit.params
        delta_train = delta_train - b1*y_train - b0
    else:
        # regress yhat on y
        fit = sm.OLS(yhat_train, sm.add_constant(y_train)).fit()
        b0, b1 = fit.params
        yhat_train = (yhat_train - b0) / b1
        delta_train = yhat_train - y_train

    train_time = time.time() - start_time

    # TEST
    start_time = time.time()
    yhat_test  = model.predict(X_test)
    if approach == 1:
        delta_test  = yhat_test - y_test
        delta_test = delta_test - b1*y_test - b0
    else:
        yhat_test = (yhat_test - b0) / b1
        delta_test = yhat_test - y_test
    test_time = time.time() - start_time

    # MAE
    mae_train = (np.abs(delta_train)).mean()
    mae_test = (np.abs(delta_test)).mean()

    # target-residual correlation
    corr_train = np.corrcoef(y_train, -delta_train)[0,1]
    corr_test = np.corrcoef(y_test, -delta_test)[0,1]
    return mae_train, mae_test, train_time, test_time, corr_train, corr_test

print(f'Starting regression loop with {n_bootstrap_iterations} bootstrap iterations')
for n in range(n_bootstrap_iterations):
    if n % 2 == 0: print('iteration', n)

    np.random.seed(42 + n)      # for reproducibility
    # bootstrap training data
    boot_ix = np.random.choice(np.arange(n_train), size=n_train, replace=True)
    X_boot, y_boot = X_train[boot_ix, :], y_train[boot_ix]

    for m in range(n_constraints): # constraints

        if m==0: # Standard Scikit-Learn models
            linreg = LinearRegression()
            ridge = GridSearchCV(Ridge(), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(KernelRidge(), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elif m==1: # zero correlation
            linreg = ccr.LinearRegression(correlation_bound=0)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elif m==2: # bounded correlation 0.1
            linreg = ccr.LinearRegression(correlation_bound=0.1)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0.1), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0.1), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elif m==3: # bounded correlation 0.2
            linreg = ccr.LinearRegression(correlation_bound=0.2)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0.2), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0.2), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elif m==4: # bounded correlation 0.3
            linreg = ccr.LinearRegression(correlation_bound=0.3)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0.3), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0.3), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elif m >= 5: # Standard approach + Post hoc correction
            linreg = LinearRegression()
            ridge = GridSearchCV(Ridge(), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(KernelRidge(), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')
            approach = m - 4
            mae[n, m, 0, 0], mae[n, m, 0, 1], times[n, m, 0, 0], times[n, m, 0, 1], corrs[n, m, 0, 0], corrs[n, m, 0, 1] = fit_model_and_correct(linreg, X_boot, y_boot, X_test, y_test, approach)
            mae[n, m, 1, 0], mae[n, m, 2, 1], times[n, m, 1, 0], times[n, m, 1, 1], corrs[n, m, 1, 0], corrs[n, m, 1, 1] = fit_model_and_correct(ridge, X_boot, y_boot, X_test, y_test, approach)
            mae[n, m, 2, 0], mae[n, m, 1, 1], times[n, m, 2, 0], times[n, m, 2, 1], corrs[n, m, 2, 0], corrs[n, m, 2, 1] = fit_model_and_correct(kr, X_boot, y_boot, X_test, y_test, approach)
            continue

        # Fit models
        mae[n, m, 0, 0], mae[n, m, 0, 1], times[n, m, 0, 0], times[n, m, 0, 1], corrs[n, m, 0, 0], corrs[n, m, 0, 1] = at.fit_model(linreg, X_boot, y_boot, X_test, y_test)
        mae[n, m, 1, 0], mae[n, m, 1, 1], times[n, m, 1, 0], times[n, m, 1, 1], corrs[n, m, 1, 0], corrs[n, m, 1, 1] = at.fit_model(ridge, X_boot, y_boot, X_test, y_test)
        mae[n, m, 2, 0], mae[n, m, 2, 1], times[n, m, 2, 0], times[n, m, 2, 1], corrs[n, m, 2, 0], corrs[n, m, 2, 1] = at.fit_model(kr, X_boot, y_boot, X_test, y_test)

# save results
pickle.dump( (times, mae, corrs), open(f'regression_results_{dataset}_train_test.pickle', 'wb' ) )
