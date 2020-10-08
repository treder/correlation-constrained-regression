# correlation-constrained-regression

Introduces correlation constraints for **Linear**, **Ridge**, and **Kernel Ridge** regression. 

In their standard form, we can formulate these 
models as unconstrained optimization problems of the form

```
minimize     L(y, yhat)
```

where `y` is the target values and `yhat` is the predictions. Correlation constrained regression is given by adding a correlation constraint and arriving at 

```
minimize     L(y, yhat)
subject to   |corr(y, e)| <= correlation_bound
```

where `corr` is Pearson correlation, `e = y - yhat` is the residuals, and correlation bound 
is a hyperparameter that controls the maximum amount of permissible correlation. 
The resultant models have been implemented in both Python and Matlab.

## Python

The module [`correlation_constrained_regression.py`](correlation_constrained_regression.py) provides three models, `LinearRegression`, `Ridge`, and `KernelRidge`.
They extend the eponymous models in [Scikit-Learn](https://scikit-learn.org/) with an additional parameter `correlation_bound` 
(a value between 0 and 1) that specifies the maximally permissible correlation between targets and residuals. 
The following code example illustrates the `LinearRegression` model:

```python
import numpy as np
import correlation_constrained_regression as ccr

# create some regression data
X = np.array([[1, 1], [1, 2], [3, 2], [3, 3], [4, 3], [4, 4]])
y = np.dot(X, np.array([1, 2])) + np.array([0.1, 0.2, -0.1, -0.2, -0.1, -0.2])

# fit correlation constrained model and calculate residual correlation
reg = ccr.LinearRegression(correlation_bound=0.01).fit(X, y)
print('corr(y, e):', np.corrcoef(y, y - reg.predict(X))[0,1])

# instead of calculating the correlation by hand, we can use the built-in method:
print('corr(y, e):', reg.calculate_residual_correlation(X, y))

# scaling factor
print('theta:', reg.theta_)

# for comparison: let's train a standard linear regression model in Scikit-Learn and print the correlation
reg = sklearn.linear_model.LinearRegression().fit(X, y)
print('corr(y, e):', np.corrcoef(y, y - reg.predict(X))[0,1])
```

Fitting `Ridge` and `KernelRidge` works analogous:

```python
ridge = ccr.Ridge(correlation_bound=0.1, alpha=10).fit(X, y)
krr = ccr.KernelRidge(correlation_bound=0, kernel='rbf', gamma=1).fit(X, y)
print('corr(y, e):', ridge.calculate_residual_correlation(X, y))
print('corr(y, e):', krr.calculate_residual_correlation(X, y))
```

You can use the models in the same way as other Scikit-Learn models. 
For instance, let us use [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
to optimize the hyperparameters of the `KernelRidge` model:

```python
tune_KernelRidge = [
  {'kernel': ['rbf'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10]},
  {'kernel': ['poly'], 'gamma': [100, 10, 1, 1e-1], 'alpha': [1e-3, 1e-2, 1e-1, 1, 10], 'degree': [2, 3, 4, 5], 'coef0':[0, 1]}
 ]
 
krr = sklearn.model_selection.GridSearchCV(ccr.KernelRidge(correlation_bound=0), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')
```

## Matlab

Regression models with correlation constraints are implemented in the [MVPA-Light](https://github.com/treder/MVPA-Light/) toolbox. 
The hyperparameter `correlation_bound` (a value between 0 and 1) specifies the maximally permissible correlation between targets and residuals. 
The following code example illustrates the Ridge regression model:


```matlab
% create some regression data
X = [1, 1; 1, 2; 3, 2; 3, 3; 4, 3; 4, 4]
y = X * [1, 2]' + [0.1, 0.2, -0.1, -0.2, -0.1, -0.2]'

% get hyperparameter struct
param = mv_get_hyperparameter('ridge');

% specify correlation bound
param.correlation_bound = 0.1;

% train model
model = train_ridge(param, X, y);

% scaling factor
fprintf('theta = %.4f\n', model.theta)
```

Linear regression is included in ridge model (set `param.lambda=0`). Fitting a kernel ridge models is analogous:

```matlab
param = mv_get_hyperparameter('kernel_ridge');
param.correlation_bound = 0.1;

model = train_kernel_ridge(param, X, y);
```

The models can be used in a cross-validation framework using the [`mv_regress`](https://github.com/treder/MVPA-Light/blob/master/mv_regress.m) function:

```matlab
cfg = [];
cfg.model  = 'ridge';
cfg.metric = 'r_squared';
cfg.hyperparameter = [];
cfg.hyperparameter.correlation_bound = 0.1;

r2 = mv_regress(cfg, X, y);
```
