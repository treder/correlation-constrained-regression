import sklearn, sklearn.linear_model, sklearn.kernel_ridge
import numpy as np

# def _calculate_intercept(self, X, y):
#     '''
#     Calculate the intercept
#     '''
#     mx = X.mean(axis=0)
#     self.intercept_ = y.mean() - mx.dot(self.coef_)


# def scale_coef(self, theta):
#     '''
#     Performs the scaling of the regression coefficients by a factor ``theta``
#     in Linear and Ridge Regression.
#     '''
#     self.coef_ = self.orig_coef_ * theta

# def scale_dual_coef(self, theta):
#     '''
#     Performs the scaling of the dual regression coefficients by a factor ``theta``
#     in Kernel Ridge Regression.
#     '''
#     self.dual_coef_ = self.orig_dual_coef_ * theta

def calculate_scaling_factor(y, yhat, correlation_bound):
    '''
    Calculates the scaling factor ``theta`` that will make the model meet the correlation constraint
    '''
    yc = y - y.mean()
    yhatc = yhat - yhat.mean()
    y_residual_correlation_uncorrected_ = yc.dot(yc - yhatc) / (np.linalg.norm(yc)*np.linalg.norm(yc - yhatc))

    if y_residual_correlation_uncorrected_ < correlation_bound:
        # dont fix it if it aint broken
        theta = 1
    elif correlation_bound == 0:
        # zero correlation constraint
        # determine scaling factor that sets corr(y, y - yhat) = 0
        theta = yc.dot(yc) / yc.dot(yhatc)
    else:
        # bounded correlation constraint
        # determine scaling factor that sets corr(y, y - yhat) to rho or -rho
        y2 = yc.dot(yc)
        yhat2 = yhatc.dot(yhatc)
        yyhat = yc.dot(yhatc)
        rho2 = correlation_bound**2
        c = yyhat**2 - rho2*y2*yhat2

        # since we use square root to solve for theta we get two solutions
        # one gives us corr(y,e) = rho, the other corr(y,e) = -rho
        tmp1 = y2 * yyhat * (1-rho2)/c - y2/c * np.sqrt( rho2 * (1-rho2) * (y2*yhat2 - yyhat**2))
        tmp2 = y2 * yyhat * (1-rho2)/c + y2/c * np.sqrt( rho2 * (1-rho2) * (y2*yhat2 - yyhat**2))
        # make sure theta with the smaller absolute theta comes first (it corresponds to positive rho)
        theta = tmp1 if np.abs(tmp1)<np.abs(tmp2) else tmp2

    return theta

def calculate_residual_correlation(self, X, y):
    '''Given features ``X`` and targets ``y``, calculates
    the correlation between ``y`` and the prediction residuals
    ``y - yhat``
    '''
    yc = y - y.mean()
    yhatc = self.predict(X)
    yhatc -= yhatc.mean()
    return yc.dot(yc - yhatc) / (np.linalg.norm(yc)*np.linalg.norm(yc - yhatc))

class LinearRegression(sklearn.linear_model.LinearRegression):
    """Ordinary least squares Linear Regression with correlation constraints.

    This class inherits from Scikit-Learn's LinearRegression. It extends
    the model with an additional parameter ``correlation_bound``. Only the
    additional parameters and attributes are described here.

    Parameters
    ----------
    correlation_bound : float, default=None
        If set to a value ``r`` with ``0 <= r <= 1`` defines a bound for the correlation
        between the residuals and y, that is, ``corr(y, y-yhat) <= r``

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Regression weight(s)
    orig_coef_ : ndarray of shape (n_features,)
        Original coefficients of the unconstrained model if correlation constraint are applied
    #y_residual_correlation_uncorrected_ : float
    #    correlation between ``y`` and the residuals ``y - yhat`` before correction
    #y_residual_correlation_corrected_ : float
    #    correlation between ``y`` and the residuals ``y - yhat`` after correction
    a_ : float
        Scaling factor such that ``a_ * coef_`` yields a model that satisfies the
        correlation constraint


    See Also
    --------
    Ridge : Correlation constrained ridge regression
    KernelRidge : Correlation constrained kernel ridge regression

    Examples
    --------
    >>> import numpy as np
    >>> import correlation_constrained_regression as ccr
    >>> X = np.array([[1, 1], [1, 2], [3, 2], [3, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + np.array([0.1, 0.2, -0.1, -0.2])
    >>> reg = ccr.LinearRegression(correlation_bound=0.01).fit(X, y)
    >>> np.corrcoef(y, y - reg.predict(X))[0,1]
    >>> reg.score(X, y)
    1.0
    >>> reg.theta_
    """
    calculate_residual_correlation = calculate_residual_correlation

    def __init__(self, correlation_bound=None, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None):
        super(LinearRegression, self).__init__(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
                 n_jobs=n_jobs)
        if correlation_bound is not None:
            assert 0 <= correlation_bound <= 1, 'correlation_bound must be in the interval [0,1]'
        self.correlation_bound = correlation_bound

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model and apply the correlation bound.
        """
        super(LinearRegression, self).fit(X, y, sample_weight)
        yhat = self.predict(X)
        self.theta_ = None
        self.orig_coef_ = self.coef_

        if self.correlation_bound is not None:
                self.theta_ = calculate_scaling_factor(y, yhat, self.correlation_bound)
                self.coef_ *= self.theta_             # scale regression coefficients
                if self.fit_intercept:
                    mx = X.mean(axis=0)
                    self.intercept_ = y.mean() - mx.dot(self.coef_) # fix intercept
        return self

class Ridge(sklearn.linear_model.Ridge):
    """Linear least squares with l2 regularization and correlation constraints.

    This class inherits from Scikit-Learn's Ridge. It extends
    the model with an additional parameter ``correlation_bound``. Only the
    additional parameters and attributes are described here.

    Parameters
    ----------
    correlation_bound : float, default=None
        If set to a value ``r`` with ``0 <= r <= 1`` defines a bound for the correlation
        between the residuals and y, that is, ``corr(y, y-yhat) <= r``

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Regression weight(s)
    orig_coef_ : ndarray of shape (n_features,)
        Original coefficients of the unconstrained model if correlation constraint are applied
    #y_residual_correlation_uncorrected_ : float
    #    correlation between ``y`` and the residuals ``y - yhat`` before correction
    #y_residual_correlation_corrected_ : float
    #    correlation between ``y`` and the residuals ``y - yhat`` after correction
    theta_ : float
        Scaling factor such that ``theta_ * coef_`` yields a model that satisfies the
        correlation constraint

    See Also
    --------
    LinearRegression : Correlation constrained OLS regression
    KernelRidge : Correlation constrained kernel ridge regression

    Examples
    --------
    >>> import numpy as np
    >>> import correlation_constrained_regression as ccr
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> model = ccr.Ridge(alpha=1.0, correlation_bound=0.1)
    >>> model.fit(X, y)
    >>> print(f'corr(y, e): uncorrected {model.y_residual_correlation_uncorrected_:.3f}, corrected {model.y_residual_correlation_corrected_:.3f}')
    """
    calculate_residual_correlation = calculate_residual_correlation

    def __init__(self, correlation_bound=None, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto", random_state=None, **kwargs):
        super(Ridge, self).__init__(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize,
                 copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver, random_state=random_state, **kwargs)
        if correlation_bound is not None:
            assert 0 <= correlation_bound <= 1, 'correlation_bound must be in the interval [0,1]'
        self.correlation_bound = correlation_bound

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model and apply the correlation bound.
        """
        super(Ridge, self).fit(X, y, sample_weight)

        yhat = self.predict(X)
        self.theta_ = None
        self.orig_coef_ = self.coef_

        if self.correlation_bound is not None:
                self.theta_ = calculate_scaling_factor(y, yhat, self.correlation_bound)
                self.coef_ *= self.theta_             # scale regression coefficients
                if self.fit_intercept:
                    mx = X.mean(axis=0)
                    self.intercept_ = y.mean() - mx.dot(self.coef_) # fix intercept
        return self

class KernelRidge(sklearn.kernel_ridge.KernelRidge):
    """Kernel Ridge Regression.

    This class inherits from Scikit-Learn's KernelRidge. It extends
    the model with an additional parameter ``correlation_bound``. Only the
    additional parameters and attributes are described here.

    Parameters
    ----------
    correlation_bound : float, default=None
        If set to a value ``r`` with ``0 <= r <= 1`` defines a bound for the correlation
        between the residuals and y, that is, ``corr(y, y-yhat) <= r``

    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Representation of weight vector(s) in kernel space
    orig_dual_coef_ : ndarray of shape (n_features,) or or (n_samples, n_targets)
        Original coefficients of the unconstrained model if correlation constraint are applied
    #y_residual_correlation_uncorrected_ : float
    #    correlation between ``y`` and the residuals ``y - yhat`` before correction
    #y_residual_correlation_corrected_ : float
    #    correlation between ``y`` and the residuals ``y - yhat`` after correction
    theta_ : float
        Scaling factor such that ``theta_ * coef_`` yields a model that satisfies the
        correlation constraint

    See Also
    --------
    LinearRegression : Correlation constrained OLS regression
    Ridge : Correlation constrained ridge regression

    Examples
    --------
    >>> import numpy as np
    >>> import correlation_constrained_regression as ccr
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> model = ccr.KernelRidge(alpha=0.5, kernel='rbf', gamma=1, correlation_bound=0.01)
    >>> model.fit(X, y)
    >>> print(f'corr(y, e): uncorrected {model.y_residual_correlation_uncorrected_:.3f}, corrected {model.y_residual_correlation_corrected_:.3f}')
    """
    calculate_residual_correlation = calculate_residual_correlation

    def __init__(self, correlation_bound=None, alpha=1, kernel="linear", gamma=None, degree=3,
                 coef0=1, kernel_params=None, **kwargs):
        super(KernelRidge, self).__init__(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree,
                 coef0=coef0, kernel_params=kernel_params, **kwargs)
        if correlation_bound is not None:
            assert 0 <= correlation_bound <= 1, 'correlation_bound must be in the interval [0,1]'
        self.correlation_bound = correlation_bound

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model and apply the correlation bound.
        """
        super(KernelRidge, self).fit(X, y, sample_weight)
        yhat = self.predict(X)
        self.theta_ = None
        self.orig_dual_coef_ = self.dual_coef_

        if self.correlation_bound is not None:
                self.theta_ = calculate_scaling_factor(y, yhat, self.correlation_bound)
                self.dual_coef_ *= self.theta_             # scale dualg regression coefficients

        return self

# print(sklearn.__version__)

# debug
# model = LinearRegression(correlation_bound=None, fit_intercept=True)
# model = LinearRegression(correlation_bound=.001, fit_intercept=True)
# model = LinearRegression(correlation_bound=0, fit_intercept=True)

# model = Ridge(correlation_bound=0.1)
# model = Ridge(correlation_bound=0)
# model = Ridge(alpha=10.01, correlation_bound=0)


# model = KernelRidge(alpha=10.01, kernel='rbf', gamma=1, correlation_bound=None)
# model = KernelRidge(alpha=10.01, kernel='rbf', gamma=1, correlation_bound=0.1)
# model = KernelRidge(alpha=10.01, kernel='rbf', gamma=1, correlation_bound=0)

# X = np.array([[1, 1], [1, 2], [3, 2], [3, 3]])
# y = np.dot(X, np.array([1, 2])) + np.array([0.1, 0.2, -0.1, -0.2])

# model = model.fit(X, y)

# # get predictions and residuals
# yh = model.predict(X)
# e = y - yh
# print('theta:', model.theta_)
# print(f'corr(y, e): corrected {model.calculate_residual_correlation(X,y)}')
