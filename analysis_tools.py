import scipy, sys, time, pickle
import numpy as np
import pandas as pd
import sklearn, sklearn.datasets

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV

import correlation_constrained_regression as ccr

def load_data(dataset):
    '''
    Loads a regression dataset

    Returns:
        X, y - features and targets
    '''
    print('Loading data')
    if dataset == 'pac2019':
        with open('pac2019_ICA_20201202_train_and_test.pickle', 'rb') as f:
            X_train, X_test, y_train, y_test, feature_names = pickle.load(f)
        print('Train:', X_train.shape, 'Test:', X_test.shape)
        return X_train, X_test, y_train, y_test, feature_names
    elif dataset == 'pac2019old':
        df = pd.read_csv('PAC2019_BrainAge_ICA_reduced.csv')

        # extract features and age
        feature_ix = [ix for ix, name in enumerate(df.columns) if name.startswith('loadings')]
        X = df.iloc[:, feature_ix].to_numpy()
        y = df['age'].to_numpy()

        print(X.shape, y.shape)
        return X, y

def fit_model(model, X_train, y_train, X_test, y_test):
    '''
    Fits model, calculates MAE on train and test data and performs timing.
    '''
    # train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # predict
    start_time = time.time()
    yhat_test  = model.predict(X_test)
    test_time = time.time() - start_time

    # target-residual correlation
    corr_train = np.corrcoef(y_train, y_train - yhat_train)[0,1]
    corr_test = np.corrcoef(y_test, y_test - yhat_test)[0,1]
#     print(f'train {corr_train:.3f}, test {corr_test:.3f}')
    return mean_absolute_error(y_train, yhat_train), mean_absolute_error(y_test, yhat_test), \
            train_time, test_time, corr_train, corr_test
