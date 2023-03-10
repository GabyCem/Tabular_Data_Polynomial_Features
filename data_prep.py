from imports import *

# Function for pre-processing the data which has combination of categorica and numerical columns
def pre_process_data(df,y_col,categoricals_default=[],split=0.2,set_seed=2047):
    
    start = time.time()
    
    # Get list of numerical and categorical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = [col for col in df.select_dtypes(include=numerics).columns 
                    if col not in [y_col]+categoricals_default]
    categorical_cols = [col for col in list(df) if col not in numeric_cols+[y_col]]
    categorical_cols = categorical_cols+categoricals_default
    
    print(f'Encoding categorical columns ',end='... ')
    # Encode categorical columns with one hot encoding
    X_cat = pd.get_dummies(df[categorical_cols])
    x_ohe_col_names = list(X_cat)

    ### --- Processing raw dataset --- ###
    # Join the processed categorical columns with numerical columns
    X_processed = pd.concat([df[numeric_cols],X_cat],axis=1)
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')
    
    # Train test split with 80:20 split
    print(f'Splitting the data ',end='... ')
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, df[y_col],test_size=split,random_state=set_seed)
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')

    # Scaling the numerical columns
    # Fit the scaler and transform train set
    print(f'Scaling the data ',end='... ')
    num_scaler_raw = StandardScaler(
        copy=True, with_mean=True, with_std=True).fit(X_train[numeric_cols])
    X_train_num_processed = pd.DataFrame(
        num_scaler_raw.transform(X_train[numeric_cols]),
        columns=numeric_cols)
    X_train_all = pd.concat(
        [X_train_num_processed.reset_index(drop=True),
         X_train[x_ohe_col_names].reset_index(drop=True)],
        axis=1,ignore_index=True)
    
    # Transform the test set
    X_test_num_processed = pd.DataFrame(
        num_scaler_raw.transform(X_test[numeric_cols]), 
        columns=numeric_cols)
    X_test_all = pd.concat([
        X_test_num_processed.reset_index(drop=True),
        X_test[x_ohe_col_names].reset_index(drop=True)],
        axis=1,ignore_index=True)
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')
        
    ### --- Construct and process polynomials of degree 2 & 3 --- ###
    # Create interaction terms where interaction is for each regressor pair + polynomial
    # Fit-transform the polynomial model for degree 2
    print(f'Fitting polynoimials of degree 2 to numericals in data ',end='... ')
    poly_deg2 = PolynomialFeatures(
        degree = 2,include_bias = False,interaction_only = False,order = 'C')
    X_poly_deg2_train = pd.DataFrame(
        poly_deg2.fit_transform(X_train[numeric_cols]),
        columns=poly_deg2.get_feature_names(input_features=numeric_cols))
    X_poly_deg2_test = pd.DataFrame(
        poly_deg2.fit_transform(X_test[numeric_cols]),
        columns=poly_deg2.get_feature_names(input_features=numeric_cols))

    # Scaling the poly fetures - degree 2
    num_scaler_poly2 = StandardScaler(
        copy=True, with_mean=True, with_std=True).fit(X_poly_deg2_train)
    X_poly_deg2_train_scaled = pd.DataFrame(
        num_scaler_poly2.transform(X_poly_deg2_train), 
        columns=X_poly_deg2_train.columns)
    X_poly_deg2_test_scaled = pd.DataFrame(
        num_scaler_poly2.transform(X_poly_deg2_test),
        columns=X_poly_deg2_test.columns)

    # Combine the scaled polynomial feautres with one hot columns
    X_train_poly2_all = pd.concat([
        X_poly_deg2_train_scaled.reset_index(drop=True),
        X_train[x_ohe_col_names].reset_index(drop=True)],
        axis=1,ignore_index=True)
    X_test_poly2_all = pd.concat([
        X_poly_deg2_test_scaled.reset_index(drop=True),
        X_test[x_ohe_col_names].reset_index(drop=True)],
        axis=1,ignore_index=True)
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')
        
    print(f'Fitting polynoimials of degree 3 to numericals in data ',end='... ')
    # Fit-transform the polynomial model for degree 3
    poly_deg3 = PolynomialFeatures(
        degree=3, include_bias=False, interaction_only=False, order='C')
    X_poly_deg3_train = pd.DataFrame(
        poly_deg3.fit_transform(X_train[numeric_cols]),
        columns=poly_deg3.get_feature_names(input_features=numeric_cols))
    X_poly_deg3_test = pd.DataFrame(
        poly_deg3.fit_transform(X_test[numeric_cols]),
        columns=poly_deg3.get_feature_names(input_features=numeric_cols))

    # Scaling the poly features - degree 3
    num_scaler_poly3 = StandardScaler(
        copy=True, with_mean=True, with_std=True).fit(X_poly_deg3_train)
    X_poly_deg3_train_scaled = pd.DataFrame(
        num_scaler_poly3.transform(X_poly_deg3_train), 
        columns=X_poly_deg3_train.columns)
    X_poly_deg3_test_scaled = pd.DataFrame(
        num_scaler_poly3.transform(X_poly_deg3_test),
        columns=X_poly_deg3_test.columns)

    # Combine the scaled polynomial feautres with one hot columns
    X_train_poly3_all = pd.concat([
        X_poly_deg3_train_scaled.reset_index(drop=True),
        X_train[x_ohe_col_names].reset_index(drop=True)],
        axis=1,ignore_index=True)
    X_test_poly3_all = pd.concat([
        X_poly_deg3_test_scaled.reset_index(drop=True),
        X_test[x_ohe_col_names].reset_index(drop=True)],
        axis=1,ignore_index=True)
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')
    
    return [
        numeric_cols,categorical_cols,x_ohe_col_names,
        y_train, y_test,
        X_train_all, X_test_all, 
        X_train_poly2_all, X_test_poly2_all,
        X_train_poly3_all, X_test_poly3_all]

# We will modify the above function to work for a dataset with no categorical columns
def pre_process_data_non_cat(df,y_col,split=0.2,set_seed=2047):
    
    start = time.time()
    
    # Get list of numerical and categorical columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = [col for col in df.select_dtypes(include=numerics).columns if col not in [y_col]]
    categorical_cols = [col for col in list(df) if col not in numeric_cols+[y_col]]

    ### --- Processing raw dataset --- ###
    
    if not len(categorical_cols) == 0:
        print(f'Encoding categorical columns ',end='... ')
        # Encode categorical columns with one hot encoding
        X_cat = pd.get_dummies(df[categorical_cols])
        x_ohe_col_names = list(X_cat)
        
        # Join the processed categorical columns with numerical columns
        X_processed = pd.concat([df[numeric_cols],X_cat],axis=1)
        print(f'{round((time.time()-start)/60,3)} mins elapsed.')
        
    else:
        X_processed = df[numeric_cols].copy()
        x_ohe_col_names = []
    
    # Train test split with 80:20 split
    print(f'Splitting the data ',end='... ')
    X_train, X_test, y_train, y_test = train_test_split(X_processed, df[y_col],test_size=split,random_state=set_seed)
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')

    # Scaling the numerical columns
    # Fit the scaler and transform train set
    print(f'Scaling the data ',end='... ')
    num_scaler_raw = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train[numeric_cols])
    X_train_num_processed = pd.DataFrame(
        num_scaler_raw.transform(X_train[numeric_cols]),
        columns=numeric_cols)
    
    if not len(categorical_cols) == 0:
        X_train_all = pd.concat(
            [X_train_num_processed.reset_index(drop=True),
             X_train[x_ohe_col_names].reset_index(drop=True)],
            axis=1,ignore_index=True)
    else:
        X_train_all = X_train_num_processed.copy()
        
    # Transform the test set
    X_test_num_processed = pd.DataFrame(
        num_scaler_raw.transform(X_test[numeric_cols]), 
        columns=numeric_cols)
    if not len(categorical_cols) == 0:
        X_test_all = pd.concat([
            X_test_num_processed.reset_index(drop=True),
            X_test[x_ohe_col_names].reset_index(drop=True)],
            axis=1,ignore_index=True)
    else: 
        X_test_all = X_test_num_processed.copy()
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')
        
    ### --- Construct and process polynomials of degree 2 & 3 --- ###
    # Create interaction terms where interaction is for each regressor pair + polynomial
    # Fit-transform the polynomial model for degree 2
    print(f'Fitting polynoimials of degree 2 to numericals in data ',end='... ')
    poly_deg2 = PolynomialFeatures(
        degree = 2,include_bias = False,interaction_only = False,order = 'C')
    X_poly_deg2_train = pd.DataFrame(
        poly_deg2.fit_transform(X_train[numeric_cols]),
        columns=poly_deg2.get_feature_names(input_features=numeric_cols))
    X_poly_deg2_test = pd.DataFrame(
        poly_deg2.fit_transform(X_test[numeric_cols]),
        columns=poly_deg2.get_feature_names(input_features=numeric_cols))

    # Scaling the poly fetures - degree 2
    num_scaler_poly2 = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_poly_deg2_train)
    X_poly_deg2_train_scaled = pd.DataFrame(
        num_scaler_poly2.transform(X_poly_deg2_train), 
        columns=X_poly_deg2_train.columns)
    X_poly_deg2_test_scaled = pd.DataFrame(
        num_scaler_poly2.transform(X_poly_deg2_test), 
        columns=X_poly_deg2_test.columns)

    # Combine the scaled polynomial feautres with one hot columns
    if not len(categorical_cols) == 0:
        X_train_poly2_all = pd.concat(
            [X_poly_deg2_train_scaled.reset_index(drop=True),
             X_train[x_ohe_col_names].reset_index(drop=True)],
            axis=1,ignore_index=True)
        X_test_poly2_all = pd.concat(
            [X_poly_deg2_test_scaled.reset_index(drop=True),
             X_test[x_ohe_col_names].reset_index(drop=True)],
            axis=1,ignore_index=True)
    else:
        X_train_poly2_all = X_poly_deg2_train_scaled.copy()
        X_test_poly2_all = X_poly_deg2_test_scaled.copy()
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')
        
    print(f'Fitting polynoimials of degree 3 to numericals in data ',end='... ')
    # Fit-transform the polynomial model for degree 3
    poly_deg3 = PolynomialFeatures(
        degree=3, include_bias=False, interaction_only=False, order='C')
    X_poly_deg3_train = pd.DataFrame(
        poly_deg3.fit_transform(X_train[numeric_cols]),
        columns=poly_deg3.get_feature_names(input_features=numeric_cols))
    X_poly_deg3_test = pd.DataFrame(
        poly_deg3.fit_transform(X_test[numeric_cols]), 
        columns=poly_deg3.get_feature_names(input_features=numeric_cols))

    # Scaling the poly fetures - degree 3
    num_scaler_poly3 = StandardScaler(
        copy=True, with_mean=True, with_std=True).fit(X_poly_deg3_train)
    X_poly_deg3_train_scaled = pd.DataFrame(
        num_scaler_poly3.transform(X_poly_deg3_train),
        columns=X_poly_deg3_train.columns)
    X_poly_deg3_test_scaled = pd.DataFrame(
        num_scaler_poly3.transform(X_poly_deg3_test),
        columns=X_poly_deg3_test.columns)

    # Combine the scaled polynomial feautres with one hot columns
    if not len(categorical_cols) == 0:
        X_train_poly3_all = pd.concat(
            [X_poly_deg3_train_scaled.reset_index(drop=True),
             X_train[x_ohe_col_names].reset_index(drop=True)],
            axis=1,ignore_index=True)
        X_test_poly3_all = pd.concat(
            [X_poly_deg3_test_scaled.reset_index(drop=True),
             X_test[x_ohe_col_names].reset_index(drop=True)],
            axis=1,ignore_index=True)
    else:
        X_train_poly3_all = X_poly_deg3_train_scaled.copy()
        X_test_poly3_all = X_poly_deg3_test_scaled.copy()
    print(f'{round((time.time()-start)/60,3)} mins elapsed.')
    
    return [
        numeric_cols,categorical_cols,x_ohe_col_names,
        y_train, y_test,
        X_train_all, X_test_all, 
        X_train_poly2_all, X_test_poly2_all,
        X_train_poly3_all, X_test_poly3_all]

# Let us write a function that can be used to calculate the variance 
# inflation factor (VIF) for the columns in a given pandas DataFrame. 

def get_vif_df(df, cols_remove, vif_tol=5.0):
    
    # VIF measures the correlation between each feature and all other features in the dataset, 
    # and is commonly used to detect multicollinearity (high correlation) between features.

    # List of numeric data types to be included in the analysis
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Create a list of all numeric columns in the input DataFrame, except for those to be removed
    numeric_cols = [col for col in df.select_dtypes(include=numerics).columns if col not in cols_remove] 
    
    # Add a constant column to the numeric columns to calculate VIF
    X_temp_vif = sm.add_constant(df[numeric_cols])
    
    # Create an empty DataFrame to store VIF values and column names
    vif_train_c = pd.DataFrame()
    
    # Calculate VIF values for each column and store in the DataFrame
    vif_train_c["VIF Factor"] = [variance_inflation_factor(X_temp_vif.values, i) for i in range(X_temp_vif.values.shape[1])]
    vif_train_c["features"] = X_temp_vif.columns
    
    # Print the shape and contents of the DataFrame for debugging
#     print(vif_train_c.shape)
#     print(vif_train_c)
    
    # Create a list of column names with VIF less than the threshold value
    vif_cols_to_keep = vif_train_c[vif_train_c["VIF Factor"]<vif_tol]['features'].values.tolist()
    
    # Return the VIF DataFrame and list of column names to keep
    return vif_train_c, vif_cols_to_keep


# Let us now wrie a function that performs feature selection on the input dataframe by 
# selecting the top 'n' features based on their F-statistic score using the f_classif 
# score function from scikit-learn's SelectKBest module.

def feature_select_df(df, y_col, n=10):
    
    # Initialize an instance of the SelectKBest class with the f_classif score function and select the top 'n' features
    fs = SelectKBest(score_func=f_classif, k=n)
    
    # Perform feature selection on the input dataframe by transforming it with the SelectKBest instance
    df_out = fs.fit_transform(df[vif_cols_to_keep], df[y_col])
    
    # Get the names of the selected features and create a new dataframe with only those features and the target column
    cols_out = list(fs.get_feature_names_out())
    df_out = pd.DataFrame(df_out, columns=cols_out)
    df_out[y_col] = df[y_col]
    
    # Return the new dataframe and the instance of the SelectKBest class used for feature selection
    return df_out, fs


# Functions for summarising the data for eda - Understanding the data
def summarize_numeric_cols(df,target_col,b_w=0.05,h=300,w=500):
    
    # print header
    print(' -- Summary of numerical columns --')
    
    # get list of all numeric columns except for the target column
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in [target_col]]
    
    # loop through each numeric column
    for col_c in numeric_cols:
        
        # create a histogram with the given column as x-axis and the target column as color
        fig = px.histogram(df,x=col_c,color=y_col_c,marginal="box")
        
        # adjust the layout of the histogram
        fig.update_layout(
            bargap=b_w, # gap between bars
            height=h, # height of the figure
            width=w, # width of the figure
            margin=go.layout.Margin(l=50, r=50, t=50, b=50), # margin around the figure
            font=dict(size=10,family="Courier New"), # font properties
            title=dict(font_size=14,text=f"Distribution of {col_c} by target",
                       font_family="Arial",y=0.99,x=0.5,xanchor='center',yanchor='top'), # title of the figure
            template='plotly_dark') # template for the figure
        
        # show the figure
        fig.show()

# pairs plot
def pair_plot(df, target_col, h=600, w=800):
    
    # get list of all numeric columns except for the target column
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in [target_col]]
    
    # Create a scatter matrix plot with given dimensions and target variable
    fig = px.scatter_matrix(df, dimensions=numeric_cols, color=target_col)
    
    # Update the layout of the plot
    fig['layout'].update(
        height=h, width=w, 
        margin=go.layout.Margin(l=50, r=50, t=50, b=50), # Set margin values
        font=dict(size=10, family="Courier New"), # Set font properties
        title=dict(font_size=14, text='Pairs plot summary', # Set title and its properties
                   font_family="Arial", y=0.99, x=0.5, xanchor='center', yanchor='top'),
        template='plotly_dark' # Set plot template
    )
    
    # Display the plot
    fig.show();