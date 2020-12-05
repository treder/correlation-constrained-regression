'''
OLD VERSION: DOES NOT USE TEST SET (USES CROSS-VALIDATION ON TRAINING SET INSTEAD)
Performs cross-validation on the training set
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
X, y = at.load_data(dataset)

tune_KernelRidge = [
  {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}
#,
#  {'kernel': ['poly'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10], 'degree': [2, 3, 4, 5], 'coef0':[0, 1]}
 ]
# tune_KernelRidge = {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}
tune_Ridge = {'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}

import time
n_iterations = 100
n_constraints = 7

# iterations x constraints (unconstrained,zero,bounded 0.1, 0.2, 0.3) x models (linear,ridge,kernelridge) x train/inference phase
times = np.zeros((n_iterations, n_constraints, 3, 2))
mae = np.zeros((n_iterations, n_constraints, 3, 2))
corrs = np.zeros((n_iterations, n_constraints, 3, 2))

def fit_model_and_correct(model, X_train, y_train, X_test, y_test, approach):
    '''
    Fits model, calculates MAE on train and test data and performs timing.
    '''
    # Step 1: Standard regression
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_test  = model.predict(X_test)

    # Step 2: Train correction coefficients on train data
    delta_train = yhat_train - y_train
    delta_test  = yhat_test - y_test
    if approach == 1:
        # regress delta on y
        fit = sm.OLS(delta_train, sm.add_constant(y_train)).fit()
        b0, b1 = fit.params
        delta_train = delta_train - b1*y_train - b0
        delta_test = delta_test - b1*y_test - b0
        mae_train = (np.abs(delta_train)).mean()
        mae_test = (np.abs(delta_test)).mean()
    else:
        # regress yhat on y
        fit = sm.OLS(yhat_train, sm.add_constant(y_train)).fit()
        b0, b1 = fit.params
        yhat_train = (yhat_train - b0) / b1
        yhat_test = (yhat_test - b0) / b1
        delta_train = yhat_train - y_train
        delta_test = yhat_test - y_test
        mae_train = mean_absolute_error(y_train, yhat_train)
        mae_test = mean_absolute_error(y_test, yhat_test)


    # target-residual correlation
    corr_train = np.corrcoef(y_train, -delta_train)[0,1]
    corr_test = np.corrcoef(y_test, -delta_test)[0,1]
    return mae_train, mae_test, corr_train, corr_test

print(f'Starting regression loop with {n_iterations} iterations')
for n in range(n_iterations):
    if n % 2 == 0: print('iteration', n)

    # get train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n)

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
            mae[n, m, 0, 0], mae[n, m, 0, 1], corrs[n, m, 0, 0], corrs[n, m, 0, 1] = fit_model_and_correct(linreg, X_train, y_train, X_test, y_test, approach)
            mae[n, m, 1, 0], mae[n, m, 2, 1], corrs[n, m, 1, 0], corrs[n, m, 1, 1] = fit_model_and_correct(ridge, X_train, y_train, X_test, y_test, approach)
            mae[n, m, 2, 0], mae[n, m, 1, 1], corrs[n, m, 2, 0], corrs[n, m, 2, 1] = fit_model_and_correct(kr, X_train, y_train, X_test, y_test, approach)
            continue

        # Fit models
        mae[n, m, 0, 0], mae[n, m, 0, 1], times[n, m, 0, 0], times[n, m, 0, 1], corrs[n, m, 0, 0], corrs[n, m, 0, 1] = at.fit_model(linreg, X_train, y_train, X_test, y_test)
        mae[n, m, 1, 0], mae[n, m, 1, 1], times[n, m, 1, 0], times[n, m, 1, 1], corrs[n, m, 1, 0], corrs[n, m, 1, 1] = at.fit_model(ridge, X_train, y_train, X_test, y_test)
        mae[n, m, 2, 0], mae[n, m, 2, 1], times[n, m, 2, 0], times[n, m, 2, 1], corrs[n, m, 2, 0], corrs[n, m, 2, 1] = at.fit_model(kr, X_train, y_train, X_test, y_test)

# save results
pickle.dump( (times, mae, corrs), open(f'regression_results_{dataset}.pickle', 'wb' ) )
