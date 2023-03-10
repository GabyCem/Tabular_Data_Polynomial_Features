from imports import *
from metrics import *

# Let us now write a function that trains and evaluates different models on a given dataset
# using Hyperopt to optimize the hyperparameters.

def run_models_on_dataset(models_to_run,X_train, y_train,max_runs_per_model = None,rand_int = 2047):
    
    # Define search space for hyperparameter tuning for each model.
    # Each model has its own set of hyperparameters and its search space is defined as a list
    # consisting of the classifier object and a dictionary with its hyperparameters.
    # The hyperparameters are defined using the hyperopt library.
    
    if not max_runs_per_model:
        max_runs_per_model = [10]*len(models_to_run)
    
    # Define the search spaces for each model
    search_space_all = {
        
        'log_reg':[
            LogisticRegression(random_state=rand_int),
            {
                'C': hp.lognormal('LR_C', 0, 1.0),
                'max_iter' : hp.choice('max_iter', range(100,1000)),
                'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
                'tol' : hp.uniform('tol', 0.00001, 0.001),
            }
        ],
        
        'elastic_net':[
            LogisticRegression(random_state=rand_int,penalty='elasticnet',solver='saga',n_jobs=-1),
            {
                'C': hp.lognormal('LR_C', 0, 10.0),
                'max_iter' : hp.choice('max_iter', range(100,1000)),
                'l1_ratio':hp.uniform('l1_ratio', 0, 1.0),
                'tol' : hp.uniform('tol', 0.000001, 0.0001),
            }
        ],
        
        'svm': [
            SVC(kernel = 'linear',random_state=rand_int),
            {
                'C': hp.uniform('C', 0.1, 20.0),
                'gamma': hp.uniform('gamma', 0, 20),
                'coef0' : hp.uniform('coef0',0.0, 10.0),
                'tol' : hp.uniform('tol', 0.00001, 0.001),
                'shrinking' : hp.choice('shrinking',[True,False])            
            }
        ],
        
        'knn':[
            KNeighborsClassifier(n_jobs=-1),
            {
                'n_neighbors': hp.choice('n_neighbors', range(1,50)),
                'leaf_size' : hp.choice('leaf_size', range(1,30)),
                'p': hp.choice('p',[1,2]),
                'weights': hp.choice('weights',['uniform','distance'])
            }
        ],
        
        'lda':[
            LinearDiscriminantAnalysis(),
            {
                'solver': hp.choice('solver',['lsqr', 'eigen']),
                'shrinkage' : hp.uniform('shrinkage', 0, 1),
                'store_covariance' : hp.choice('store_covariance',[True,False])
            }
        ],
        
        'qda':[
            QuadraticDiscriminantAnalysis(),
            {
                'reg_param': hp.uniform('reg_param',0.00001,0.5), 
                'store_covariance': hp.choice('store_covariance', [True, False]),
                'tol' : hp.uniform('tol', 0.000001, 0.0001),    
            }
        ],
        
        'naive_bayes':[
            GaussianNB(),
            {
                'var_smoothing': hp.uniform('var_smoothing',1e-9,1e-5)
            }
            ],
        
        'xgb':[
            XGBClassifier(objective='binary:logistic',eval_metric='auc',seed=rand_int),
            {
                'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
                'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
                'max_depth':  hp.choice('max_depth', np.arange(2, 10, dtype=int)),
                'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            }
        ]
    }

    rf_space = {
        'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 2, 12, 1),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.quniform('n_estimators', 100,1500,50)
    }

    def objective(model_params):
        
        # Extract the classifier and its hyperparameters from the `model_params` dictionary.
        clf = model_params[0]
        clf.set_params(**model_params[1])  # set the hyperparameters of the classifier
    
        # Use cross-validation to estimate the accuracy of the classifier on the training data.
        # `cv=5` specifies 5-fold cross-validation, and `n_jobs=-1` uses all available CPU cores.
        score = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1, error_score=9999).mean()
    
        # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
        # `status` is a flag indicating the status of the optimization, which is set to `STATUS_OK` if the function
        # completes successfully.
        
        return {'loss': -score, 'status': STATUS_OK}


    def hyperopt_objective_rf(space,data):
        
        # Create a RandomForestClassifier object with hyperparameters specified by the `space` dictionary
        clf = RandomForestClassifier(
            criterion=space['criterion'], 
            max_depth=space['max_depth'],
            max_features=space['max_features'],
            min_samples_leaf=space['min_samples_leaf'],
            min_samples_split=space['min_samples_split'],
            n_estimators=int(space['n_estimators']), 
            random_state=2047
        )
        
        # Compute the cross-validation score for the RandomForestClassifier object using X_train and y_train
        score = cross_val_score(clf, X=data[0], y=data[1], cv=5, n_jobs=-1, error_score=9999).mean()
        # Return a dictionary containing the negative of the score (because fmin() 
        # tries to minimize the objective) and the status of the optimization

        return {'loss': -score, 'status': STATUS_OK}
    
    # Run all the models with tuning for the dataset at hand
    best_params_reservoir = {}
    start = time.time()
    for i, model in enumerate(models_to_run):
    
        print(f'Running parameter tuning for model - {model} ',end='...')
        # Check if the current model is not Random Forest (RF)
        if not model == 'rf':

            # Get the search space for the current model from the dictionary
            search_space_current = search_space_all[model]

            # Initialize a Hyperopt Trials object to keep track of the optimization process
            trials = Trials()

            # Find the best hyperparameters for the current model using Hyperopt's Tree-structured Parzen Estimator (TPE) algorithm
            best_params = fmin(objective, search_space_current, algo=tpe.suggest, max_evals=max_runs_per_model[i], trials=trials)

            # Convert the best hyperparameters from the search space to the original space and store them in the dictionary
            best_params_original_space = space_eval(search_space_current, best_params)
            best_params_reservoir[model] = best_params_original_space

            # Print a message to indicate that the tuning for the current model has completed and the elapsed time
            print(f'{round((time.time()-start)/60,3)} mins elapsed.')
    
        # If the current model is Random Forest (RF)
        else:

            # Initialize a Hyperopt Trials object to keep track of the optimization process
            trials = Trials()

            # Find the best hyperparameters for the RF model using Hyperopt's TPE algorithm
            # best = fmin(fn= hyperopt_objective_rf, space= rf_space, algo= tpe.suggest, max_evals = max_runs_per_model[i], trials= trials)
            best_partial = partial(hyperopt_objective_rf, data=[X_train,y_train])
            best = fmin(fn = best_partial,space = rf_space,algo= tpe.suggest, max_evals = max_runs_per_model[i], trials= trials)

            # Convert the best hyperparameters from the search space to the original space and store them in the dictionary
            best_params_original_space = space_eval(rf_space, best)
            best_params_reservoir[model] = best_params_original_space

            # Print a message to indicate that the tuning for the RF model has completed and the elapsed time
            print(f'{round((time.time()-start)/60,3)} mins elapsed.')
            
    return best_params_reservoir