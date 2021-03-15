
% Load data and extract features and target
% df = readtable('PAC2019_BrainAge_ICA_reduced.csv');
df = readtable('pac2019_ICA_20201202.csv');

% keep only train subjects
df = df(df.holdout==0, :);

y = df{:, 'age'};
df{:, 9:end} = zscore(df{:, 9:end});

% get hyperparameter structs
param_linreg = mv_get_hyperparameter('ridge');
param_linreg.lambda = 0;

param_ridge = mv_get_hyperparameter('ridge');
param_ridge.lambda =  100;

rho = [0, 0.1, 0.2, 0.3];
a = cell(2, 2, numel(rho));  % matter (GM/WM) x model x rho

models = cell(2, 2, numel(rho)); 

for m = 1:2
    
    if m == 1
        ix = contains(df.Properties.VariableNames, 'loadings_gm'); % grey matter voxels
    else    
        ix = contains(df.Properties.VariableNames, 'loadings_wm'); % white matter voxels
    end
    Xtrain = df{:, ix};
    
    for r = 1:numel(rho)
        
        param_linreg.correlation_bound = rho(r);
        param_ridge.correlation_bound = rho(r);
        
        % train models
        models{m,1,r} = train_ridge(param_linreg, Xtrain, y); % OLS
        models{m,2,r} = train_ridge(param_ridge, Xtrain, y);  % ridge
        
        % activation pattern
        C = cov(Xtrain);
        a{m,1,r} = C * models{m,1,r}.w;
        a{m,2,r} = C * models{m,2,r}.w;
    end
end

save('activation_patterns_pac2019.mat', 'a','rho','df','models');

