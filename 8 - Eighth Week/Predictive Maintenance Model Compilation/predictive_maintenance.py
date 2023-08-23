"""
Author: Isaac Monroy
Title: Predictive Maintenance Model Compilation
Description:
    The objective of this algorithm is to be able to
    offer the user versatility in regards to choosing
    a model that will benefit and ultimately predict 
    accurately a time series sensor dataset for the 
    goal of providing predictive maintenance. 
    
    The reason for selecting the chosen models, LSTM, 
    Random Forest Regressor, and XGBRegressor is 
    because they were tested with different time-series
    datasets and they provided the best prediction 
    results compared to other models.
"""
import numpy as np # For numerical operations
import pandas as pd # Data manipulation and analysis

# Keras modules for building, training, and evaluating the LSTM model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

# Scikit-learn modules
# Enable the application of different preprocessing steps to subsets of features, simplifying preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
# Handle missing values in the data, improving model robustness
from sklearn.impute import SimpleImputer
# Metrics for model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
# GridSearchCV for hyperparameter tuning, train_test_split for creating training and test datasets
from sklearn.model_selection import GridSearchCV, train_test_split
# Chains multiple data preprocessing steps together, simplifying the preprocessing workflow
from sklearn.pipeline import Pipeline
# OneHotEncoder for categorical encoding, StandardScaler for feature scaling
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# XGBoost module
from xgboost import XGBRegressor

def preprocess_data(X_train, X_test):
    """
    Offer the handling of different feature types, modularity, 
    and easy integration with model training.
    """
    # Identify columns with missing values
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Numeric transformer: impute missing values and scale the features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer: impute missing values and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit the preprocessor on the training data and transform both the training and test data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed

def train_and_evaluate_model(model_type, X_train, y_train, X_test, y_test, epochs=100, batch_size=128):
    """
    Provide the user with a preset architecture for each model 
    offering a starting point depending on the selection for 
    the model.
    """
    if model_type == 'LSTM':
        seq_length, nb_features = X_train.shape[1], X_train.shape[2]  
        model = Sequential()
        model.add(LSTM(units=100, input_shape=(seq_length, nb_features), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='linear'))
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1, callbacks=[early_stopping])

    elif model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Predict and calculate mean absolute error
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return model, train_mae, test_mae

def grid_search_random_forest(X_train, y_train):
    """
    Provide hyperparameter tuning utilizing Gridsearch for 
    the following models to obtain optimal results for the 
    dataset.
    """
    model = RandomForestRegressor(random_state=42)

    # Define the hyperparameter search space
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Perform the Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_

def grid_search_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7, 1],
        'colsample_bytree': [0.5, 0.7, 1],
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_

def perform_grid_search(model_type, X_train, y_train):
    if model_type == "RandomForest":
        return grid_search_random_forest(X_train, y_train)
    elif model_type == "XGBoost":
        return grid_search_xgboost(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_lagged_features(X_train, X_test, max_lag, rolling_window_size):
    """
    Apply lagged features to represent and capture previous values 
    of a variable at different points in time, and apply feature
    engineering for new features that can provide more information 
    about the different patterns in the data.
    """
    X_train_lagged = X_train.copy()
    X_test_lagged = X_test.copy()
    
    # Create lagged features for each column up to the specified maximum lag
    for col in X_train.columns:
        for lag in range(1, max_lag + 1):
            X_train_lagged[f"{col}_lag_{lag}"] = X_train[col].shift(lag)
            X_test_lagged[f"{col}_lag_{lag}"] = X_test[col].shift(lag)
        
        # Create rolling window features for each column up to the specified window size
        for window_size in range(1, rolling_window_size + 1):
            X_train_lagged[f"{col}_rolling_mean_{window_size}"] = X_train[col].rolling(window=window_size).mean()
            X_test_lagged[f"{col}_rolling_mean_{window_size}"] = X_test[col].rolling(window=window_size).mean()

    # Drop any rows with NaN values created by the lagged features and rolling window functions
    X_train_lagged.dropna(inplace=True)
    X_test_lagged.dropna(inplace=True)

    return X_train_lagged, X_test_lagged

def gen_X_pp_train_test(X_train, y_train, X_test, y_test, max_lag=5, rolling_window_size=5):
    # Create lagged features for models
    X_train_lagged, X_test_lagged = create_lagged_features(X_train, X_test, max_lag, rolling_window_size)

    # Remove corresponding rows in y_train and y_test
    y_train = y_train[X_train_lagged.index]
    y_test = y_test[X_test_lagged.index]

    # Preprocess the lagged data
    X_train_lagged_preprocessed, X_test_lagged_preprocessed = preprocess_data(X_train_lagged, X_test_lagged)
    
    return X_train_lagged_preprocessed, y_train, X_test_lagged_preprocessed, y_test

# Load your dataset as a CSV file
data = pd.read_csv('sensor_data.csv')

# Separate and assign features to (X) and assign label to (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training and testing data
X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test)

# Choose from 'LSTM', 'RandomForest', 'XGBoost', or 'no_model'
model_type = 'no_model'
train_mae, test_mae = None, None

# User's choice: 'RandomForest', 'XGBoost', or 'no_grid_search'
grid_search_type = 'no_grid_search'
best_model, best_params, best_mae = None, None, None

# Ensure only one option (model or grid search) is chosen.
if model_type != 'no_model' and grid_search_type == 'no_grid_search':
    
    if model_type == 'LSTM':
        # Reshape data to 3D array (samples, time steps, features) for LSTM
        X_train_3d = X_train_preprocessed.reshape((X_train_preprocessed.shape[0], 1, X_train_preprocessed.shape[1]))
        X_test_3d = X_test_preprocessed.reshape((X_test_preprocessed.shape[0], 1, X_test_preprocessed.shape[1]))

        # Run the LSTM model with reshaped data
        model, train_mae, test_mae = train_and_evaluate_model(model_type, X_train_3d, y_train, X_test_3d, y_test)

    else:
        # Create lagged features for Random Forest or XGBoost models
        X_train_lagged_preprocessed, y_train, X_test_lagged_preprocessed, y_test = gen_X_pp_train_test(X_train, y_train, X_test, y_test)

        # Run the selected model with lagged data
        model, train_mae, test_mae = train_and_evaluate_model(model_type, X_train_lagged_preprocessed, y_train, X_test_lagged_preprocessed, y_test)

elif grid_search_type != 'no_grid_search' and model_type == 'no_model':
    # Create lagged features for grid search
    X_train_lagged_preprocessed, y_train, _, _ = gen_X_pp_train_test(X_train, y_train, X_test, y_test)

    # Perform grid search with lagged data
    model, best_params, best_mae = perform_grid_search(grid_search_type, X_train_lagged_preprocessed, y_train)

elif model_type == 'no_model' and grid_search_type == 'no_grid_search':
    print("Please choose a model type or grid search type.")

print("\nModel type metrics")
print(f"{model_type} - Training Mean Absolute Error: {train_mae}")
print(f"{model_type} - Test Mean Absolute Error: {test_mae}")

print("\nGrid search type metrics")
print("Best model parameters:")
print(best_params)
print("Best Mean Absolute Error:")
print(best_mae)

def evaluate_model_mae_mse(model, X_test, y_test, model_type):
    # If the model is an LSTM, reshape the input data to match the LSTM input requirements
    if model_type == 'LSTM':
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape(y_pred.shape[0])
    else:
        # For non-LSTM models (Random Forest and XGBoost), use the model to make predictions
        y_pred = model.predict(X_test)
    
    # Calculate mean absolute error (mae) and mean squared error (mse) between the predicted and true values
    mae_score = mean_absolute_error(y_test, y_pred)
    mse_score = mean_squared_error(y_test, y_pred)

    return mae_score, mse_score

mae_score, mse_score = evaluate_model_mae_mse(model, X_test_lagged_preprocessed, y_test[X_test_lagged.index], model_type)

print(f"Mean Absolute Error: {mae_score}")
print(f"Mean Squared Error: {mse_score}")