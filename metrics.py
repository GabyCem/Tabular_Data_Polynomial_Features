from imports imoprt *

# Let us write a function that takes in the best parameters obtained from a model selection 
# process, along with training and testing data, and computes performance metrics 
# for each of the specified models.

def make_metrics(best_params_raw_data, X_train_all, y_train, X_test_all, y_test, run_identifier):
    metrics_all = []
    models_ran = list(best_params_raw_data.keys())

    for model in models_ran:
        
        # Retrieve the best model and fit on the data
        if not model == 'rf':
            
            # Get the best model for the current iteration
            model_c = best_params_raw_data[model][0] 
            
            # Get the best hyperparameters for the current iteration
            params_c = best_params_raw_data[model][1] 
            
        else:
            
            # Create a new random forest model with default hyperparameters
            model_c = RandomForestClassifier(random_state=2047) 
            
            # Get the best hyperparameters for the current iteration
            params_c = best_params_raw_data[model] 

        # Update the model with the best hyperparameters
        if 'n_estimators' in list(params_c.keys()):
            params_c['n_estimators'] = int(params_c['n_estimators'])
        model_c.set_params(**params_c)

        # Fit the model on the training data
        model_c.fit(X_train_all, y_train)

        # Make predictions and metrics
        # Predict the target variable using the fitted model
        y_pred_c = model_c.predict(X_test_all)
        
        # Compute classification report for the predictions
        met_report = pd.DataFrame(classification_report(y_test, y_pred_c, output_dict=True)) 
        
        # Extract accuracy, precision, recall, and f1-score metrics from the classification report
        metrics_req = pd.DataFrame(
            [met_report['accuracy'][0]]+met_report['macro avg'].values[:3].tolist(),
            index=['accuracy', 'precision', 'recall', 'f1-score']).T 
        
        # Add the model name as a column to the metrics DataFrame
        metrics_req['model'] = model
        
        # Append the metrics for the current model to the list of metrics for all models
        metrics_all.append(metrics_req) 

    # Concatenate the metrics for all models into a single DataFrame
    metrics_all_final = pd.concat(metrics_all) 
    
    # Add experiment identifier as a column to the metrics DataFrame
    metrics_all_final['Exp Identifier'] = run_identifier 

    return metrics_all_final
