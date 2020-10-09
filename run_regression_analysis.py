import scipy, sys
import numpy as np
import pandas as pd
import sklearn, sklearn.datasets

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV

sys.path.append('/home/matthias/mt03/python_tools')
import file_tools as ft
import correlation_constrained_regression as ccr

def load_data(dataset):
    '''
    Loads a regression dataset

    Returns:
        X, y - features and targets
    '''
    print('Loading data')
    if dataset == 'pac2019':
        df = pd.read_csv(datadir + 'PAC2019_BrainAge_ICA.csv')

        # extract features and age
        feature_ix = [ix for ix, name in enumerate(df.columns) if name.startswith('loadings')]
        X = df.iloc[:, feature_ix].to_numpy()
        y = df['age'].to_numpy()

    print(X.shape, y.shape)
    return X, y

def fit_model(model, X_train, y_train, X_test, y_test):
    '''
    Fits model, calculates MSE on train and test data and performs timing.
    '''
    # train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # predict
    start_time = time.time()
    yhat_train = model.predict(X_train)
    yhat_test  = model.predict(X_test)
    test_time = time.time() - start_time

    # target-residual correlation
    corr_train = np.corrcoef(y_train, y_train-yhat_train)[0,1]
    corr_test = np.corrcoef(y_test, y_test-yhat_test)[0,1]
    
    return mean_absolute_error(y_train, yhat_train), mean_absolute_error(y_test, yhat_test), \
            train_time, test_time, corr_train, corr_test

dataset = 'pac2019'
X, y = load_data(dataset)

tune_KernelRidge = [
  {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]},
  {'kernel': ['poly'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10], 'degree': [2, 3, 4, 5], 'coef0':[0, 1]}
 ]
# tune_KernelRidge = {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}
tune_Ridge = {'alpha': [1e-3, 1e-2, 1e-1, 1, 10]}

import time
n_iterations = 100
n_constraints = 5

# iterations x constraints (unconstrained,zero,bounded 0.1, 0.2, 0.3) x models (linear,ridge,kernelridge) x train/inference phase
times = np.zeros((n_iterations, n_constraints, 3, 2))
mse = np.zeros((n_iterations, n_constraints, 3, 2))
corrs = np.zeros((n_iterations, n_constraints, 3, 2))

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

        # Fit models
        mse[n, m, 0, 0], mse[n, m, 0, 1], times[n, m, 0, 0], times[n, m, 0, 1], corrs[n, m, 0, 0], corrs[n, m, 0, 1] = fit_model(linreg, X_train, y_train, X_test, y_test)
        mse[n, m, 1, 0], mse[n, m, 1, 1], times[n, m, 1, 0], times[n, m, 1, 1], corrs[n, m, 1, 0], corrs[n, m, 1, 1] = fit_model(ridge, X_train, y_train, X_test, y_test)
        mse[n, m, 2, 0], mse[n, m, 2, 1], times[n, m, 2, 0], times[n, m, 2, 1], corrs[n, m, 2, 0], corrs[n, m, 2, 1] = fit_model(kr, X_train, y_train, X_test, y_test)

# save results
pickle.dump( (times, mse, corrs), open(f'regression_results_{dataset}.pickle', 'wb' ) )

