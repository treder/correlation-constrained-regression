function run_regression_analysis()

% Load data and extract features and target
df = readtable('PAC2019_BrainAge_PCA_reduced.csv');

X = df{:, 2:end};
y = df{:, 'age'};

% get hyperparameter structs
param_linreg = mv_get_hyperparameter('ridge');
param_linreg.lambda = 0;

param_ridge = mv_get_hyperparameter('ridge');
param_ridge.lambda =  [1e-3, 1e-2, 1e-1, 1, 10];

param_kernel_ridge = mv_get_hyperparameter('kernel_ridge');
param_kernel_ridge.kernel = 'rbf';
param_kernel_ridge.gamma = [100, 10, 1, 1e-1];
param_kernel_ridge.lambda = [1e-3, 1e-2, 1e-1, 1, 10];

n_iterations = 100;
n_constraints = 5;

% iterations x constraints (unconstrained,zero,bounded 0.1, 0.2, 0.3) x models (linear,ridge,kernelridge) x train/inference phase
times = zeros(n_iterations, n_constraints, 3, 2);
mse = zeros(n_iterations, n_constraints, 3, 2);
corrs = zeros(n_iterations, n_constraints, 3, 2);

fprintf('Starting regression loop with %d iterations\n', n_iterations)

for n = 1:n_iterations
    if mod(n,2) == 0, print('iteration', n), end
    
    % get train and test data
    CV = mv_get_crossvalidation_folds('holdout', y, [], [], 0.2);
    X_train = X(CV.training, :);
    X_test = X(CV.test, :);
    y_train = y(CV.training);
    y_test = y(CV.test);
    
    for m = 1:n_constraints
    
        if m==0 % Standard Scikit-Learn models
            param_linreg.correlation_bound = [];
            param_ridge.correlation_bound = [];
            param_kernel_ridge.correlation_bound = [];
            
        elseif m==1 % zero correlation
            linreg = ccr.LinearRegression(correlation_bound=0)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elseif m==2 % bounded correlation 0.1
            linreg = ccr.LinearRegression(correlation_bound=0.1)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0.1), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0.1), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elseif m==3 % bounded correlation 0.2
            linreg = ccr.LinearRegression(correlation_bound=0.2)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0.2), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0.2), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')

        elseif m==4 % bounded correlation 0.3
            linreg = ccr.LinearRegression(correlation_bound=0.3)
            ridge = GridSearchCV(ccr.Ridge(correlation_bound=0.3), param_grid=tune_Ridge, scoring='neg_mean_squared_error')
            kr = GridSearchCV(ccr.KernelRidge(correlation_bound=0.3), param_grid=tune_KernelRidge, scoring='neg_mean_squared_error')
        end
        
        % Fit models
        mse(n,m,0,0), mse(n,m,0,1), times(n,m,0,0), times(n,m,0,1), corrs(n,m,0,0), corrs(n,m,0,1) = fit_model(linreg, X_train, y_train, X_test, y_test);
        mse(n,m,1,0), mse(n,m,1,1), times(n,m,1,0), times(n,m,1,1), corrs(n,m,1,0), corrs(n,m,1,1) = fit_model(ridge, X_train, y_train, X_test, y_test);
        mse(n,m,2,0), mse(n,m,2,1), times(n,m,2,0), times(n,m,2,1), corrs(n,m,2,0), corrs(n,m,2,1) = fit_model(kr, X_train, y_train, X_test, y_test);
    end
end


save('regression_results_pac2019.mat', times, mse, corrs);



    function  [mae_train, mae_test, train_time, test_time, corr_train, corr_test] = fit_model(train_fun, test_fun, X_train, y_train, X_test, y_test):
        % Fits model, calculates MSE on train and test data and performs timing.
        % train
        tic
        model = train_fun(X_train, y_train);
        train_time = toc;

        % predict
        tic
        yhat_train = test_fun(model, X_train)
        yhat_test  = test_fun(model, X_test);
        test_time = toc;

        % target-residual correlation
        corr_train = corr(y_train, y_train-yhat_train)
        corr_test = corr(y_test, y_test-yhat_test)

        return mean_absolute_error(y_train, yhat_train), mean_absolute_error(y_test, yhat_test), \
                train_time, test_time, corr_train, corr_test
    end
    
end