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
mae = zeros(n_iterations, n_constraints, 3, 2);
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
    
        if m==1 % Standard Scikit-Learn models
            param_linreg.correlation_bound = [];
            param_ridge.correlation_bound = [];
            param_kernel_ridge.correlation_bound = [];
            
        elseif m==2 % zero correlation
            param_linreg.correlation_bound = 0;
            param_ridge.correlation_bound = 0;
            param_kernel_ridge.correlation_bound = 0;

        elseif m==3 % bounded correlation 0.1
            param_linreg.correlation_bound = 0.1;
            param_ridge.correlation_bound = 0.1;
            param_kernel_ridge.correlation_bound = 0.1;

        elseif m==4 % bounded correlation 0.2
            param_linreg.correlation_bound = 0.2;
            param_ridge.correlation_bound = 0.2;
            param_kernel_ridge.correlation_bound = 0.2;

        elseif m==5 % bounded correlation 0.3
            param_linreg.correlation_bound = 0.3;
            param_ridge.correlation_bound = 0.3;
            param_kernel_ridge.correlation_bound = 0.3;
        end
        
        % Fit models
        [mae(n,m,1,1), mae(n,m,1,2), times(n,m,1,1), times(n,m,1,2), corrs(n,m,1,1), corrs(n,m,1,2)] = fit_model(@train_ridge, param_linreg, @test_ridge, X_train, y_train, X_test, y_test);
        [mae(n,m,2,1), mae(n,m,2,2), times(n,m,2,1), times(n,m,2,2), corrs(n,m,2,1), corrs(n,m,2,2)] = fit_model(@train_ridge, param_ridge, @test_ridge, X_train, y_train, X_test, y_test);
        [mae(n,m,3,1), mae(n,m,3,2), times(n,m,3,1), times(n,m,3,2), corrs(n,m,3,1), corrs(n,m,3,2)] = fit_model(@train_kernel_ridge, param_kernel_ridge, @test_kernel_ridge, X_train, y_train, X_test, y_test);
    end
end


save('regression_results_pac2019.mat', times, mae, corrs);



    function  [mae_train, mae_test, train_time, test_time, corr_train, corr_test] = fit_model(train_fun, param, test_fun, X_train, y_train, X_test, y_test)
        % Fits model, calculates MAE on train and test data and performs timing.
        % train
        tic
        model = train_fun(param, X_train, y_train);
        train_time = toc;

        % predict
        tic
        yhat_train = test_fun(model, X_train);
        yhat_test  = test_fun(model, X_test);
        test_time = toc;

        % target-residual correlation
        corr_train = corr(y_train, y_train-yhat_train);
        corr_test = corr(y_test, y_test-yhat_test);
    
        % error estimates
        mae_train = mean_absolute_error(y_train, yhat_train);
        mae_test =  mean_absolute_error(y_test, yhat_test);
    end

    function mae = mean_absolute_error(y, yhat)
        mae = mean(abs(y-yhat));
    end
    
end