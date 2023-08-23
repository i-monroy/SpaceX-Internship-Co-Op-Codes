"""
Author: Isaac Monroy
Title: Predicting Remaining Useful Life (RUL) Algorithm
Description:
    This algorithm aims to predict the Remaining Useful Life (RUL) of 
    engines based on sensor data. By using machine learning techniques
    to generate a model that can estimate the time before a given engine
    requires maintenance or replacement.

    The process includes preprocessing steps such as removing constant 
    sensor readings, dropping highly correlated features, calculating 
    the RUL, and removing outliers from the training data.

    It uses backward stepwise regression for feature selection and the 
    Gradient Boosting Regressor for the training of the model.

    The model's performance is then evaluated using metrics like R-squared
    error, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and 
    Mean Squared Log Error (MSLE). A plot of the actual RUL values against
    the predicted RUL values is also provided for visual representation of
    the model's performance.
"""

# Data manipulation and mathematical computation libraries
import numpy as np  # Numerical computations
import pandas as pd  # Dataframe manipulation
import statsmodels.api as sm  # Statistical modeling
import time  # Time related tasks
from scipy.stats import zscore  # Z-score for outlier detection

# Machine Learning related libraries
from sklearn.model_selection import train_test_split  # Split data into training and testing sets
from sklearn.preprocessing import MinMaxScaler  # Scale data
from sklearn.ensemble import GradientBoostingRegressor  # Gradient Boosting Regressor model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error  # Error metrics

# Visualization libraries
import matplotlib.pyplot as plt  # Basic plotting functionality
import seaborn as sns  # Advanced plotting functionality

# Define feature names for our dataset
index_names = ['engine', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names=[ "(Fan inlet temperature) (◦R)",
"(LPC outlet temperature) (◦R)",
"(HPC outlet temperature) (◦R)",
"(LPT outlet temperature) (◦R)",
"(Fan inlet Pressure) (psia)",
"(bypass-duct pressure) (psia)",
"(HPC outlet pressure) (psia)",
"(Physical fan speed) (rpm)",
"(Physical core speed) (rpm)",
"(Engine pressure ratio(P50/P2)",
"(HPC outlet Static pressure) (psia)",
"(Ratio of fuel flow to Ps30) (pps/psia)",
"(Corrected fan speed) (rpm)",
"(Corrected core speed) (rpm)",
"(Bypass Ratio) ",
"(Burner fuel-air ratio)",
"(Bleed Enthalpy)",
"(Required fan speed)",
"(Required fan conversion speed)",
"(High-pressure turbines Cool air flow)",
"(Low-pressure turbines Cool air flow)" ]
col_names = index_names + setting_names + sensor_names

def load_data():
    """
    Load the training, testing, and Remaining Useful Life (RUL) data.
    The data is read from text files and returned as pandas dataframes.
    """
    # Load data from the specified paths
    df_train = pd.read_csv(('./CMaps/train_FD001.txt'), sep='\s+', header=None, names=col_names)
    df_test = pd.read_csv(('./CMaps/test_FD001.txt'), sep='\s+', header=None, names=col_names)
    df_test_RUL = pd.read_csv(('./CMaps/RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])
    return df_train, df_test, df_test_RUL

def remove_outliers(df, threshold=3):
    """
    Remove the outliers from the dataset by identifying the data points 
    that have a z-score greater than the specified threshold.
    """
    # Calculate the Z-scores of the dataframe
    z_scores = np.abs(zscore(df))
    
    # Identify the outliers
    outliers = np.where(z_scores > threshold)
    
    # Remove the outliers from the dataframe
    df = df[(z_scores < threshold).all(axis=1)]
    return df

def drop_const_values(df_train, df_test):
    """
    Drop columns in the train and test datasets that have constant values.
    """
    # List of sensor features that have constant values
    sens_const_values = [feature for feature in setting_names + sensor_names if df_train[feature].min() == df_train[feature].max()]
    
    # Remove these columns from the dataframes
    df_train.drop(sens_const_values, axis=1, inplace=True)
    df_test.drop(sens_const_values, axis=1, inplace=True)
    return df_train, df_test

def drop_highly_corr(df_train, df_test):
    """
    Drop columns in the train and test datasets that are highly correlated 
    with other columns.
    """
    # Compute the correlation matrix of the dataframe
    cor_matrix = df_train.corr().abs()
    
    # Get the upper triangle of the correlation matrix
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    
    # Identify features which are correlated more than 0.95
    corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    # Drop these columns from the dataframes
    df_train.drop(corr_features, axis=1, inplace=True)
    df_test.drop(corr_features, axis=1, inplace=True)
    return df_train, df_test

def calculate_rul(df_train):
    """
    Calculate the Remaining Useful Life (RUL) for each row in the train 
    dataset. The RUL is calculated as the total life of the engine minus 
    the current cycle number, and it is capped at 125 cycles.
    """
    # Calculate total life for each engine
    df_train_RUL = df_train.groupby(['engine']).agg({'cycle':'max'}).rename(columns={'cycle':'life'})
    
    # Join life column to the main dataframe
    df_train = df_train.merge(df_train_RUL, on=['engine'], how='left')
    
    # Calculate RUL as life minus current cycle number
    df_train['RUL'] = df_train['life'] - df_train['cycle']
    
    # Drop the life column as we no longer need it
    df_train.drop(['life'], axis=1, inplace=True)
    
    # Cap the RUL at 125 cycles
    df_train['RUL'][df_train['RUL']>125] = 125
    return df_train

def preprocess_data(df_train, df_test):
    """
    Preprocess the training and testing data by removing constant values, 
    dropping highly correlated features, calculating the Remaining Useful 
    Life (RUL), and removing outliers.
    """
    # Drop the sensors with constant values
    df_train, df_test = drop_const_values(df_train, df_test)

    # Drop all but one of the highly correlated features
    df_train, df_test = drop_highly_corr(df_train, df_test)

    # Define the maximum life of each engine, to obtain the RUL at each point in time of the engine's life 
    df_train = calculate_rul(df_train)

    # Remove outliers from the training data
    df_train = remove_outliers(df_train)

    return df_train, df_test

def backward_regression(X, y, threshold_out=0.05):
    """
    Performs backward stepwise regression on X and y by starting with all 
    columns in X and iteratively removes the feature with the highest 
    p-value greater than threshold_out.
    """
    # List of currently included features
    included = list(X.columns)
    while True:
        # Fit the OLS model
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        
        # Get the p-values of the features
        pvalues = model.pvalues.iloc[1:]
        
        # Get the max p-value
        worst_pval = pvalues.max()
        
        # If the max p-value is greater than the threshold, remove the corresponding feature
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
        else:
            # Stop the loop if no p-values are above the threshold
            break
    return included

def select_features(df_train):
    """
    Perform feature selection on the train dataset. Uses backward stepwise
    regression to select features.
    """
    # Select all columns except for engine and RUL
    X = df_train.iloc[:,1:-1]
    # Select the RUL column as the target
    y = df_train.iloc[:,-1]
    # Perform backward stepwise regression
    selected_features = backward_regression(X, y)
    return selected_features

def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Regressor model on X_train and y_train. Print
    the training time.
    """
    start = time.time()
    # Fit the model to the data
    model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    end = time.time()
    print(f"Training time: {end - start}")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the model on the test data. Prints the
    R^2, RMSE, MAE, and MSLE.
    """
    # Get the model predictions
    predictions = model.predict(X_test)
    print('R-squared error: ' + "{:.2%}".format(model.score(X_test, y_test)))
    print('Root Mean Squared Error: ' + "{:.2f}".format(mean_squared_error(y_test, predictions, squared=False)))
    print('Mean Absolute Error: ' + "{:.2f}".format(mean_absolute_error(y_test, predictions)))
    print('Mean Squared Log Error: ' + "{:.2f}".format(mean_squared_log_error(y_test, predictions)))
    
def plot_actual_vs_prediction(y_test, y_predictions):
    """
    Plot the actual RUL values against the predicted RUL values.
    """
    plt.style.use('seaborn-white')
    plt.rcParams['figure.figsize']=20,5 
    fig,ax = plt.subplots()
    plt.ylabel('RUL')
    plt.xlabel('Engine nr')

    g = sns.lineplot(x = np.arange(0,len(y_test)),
                    y=y_test,
                    color='gray',
                    label = 'actual',
                    ax=ax)

    f = sns.lineplot(x = np.arange(0,len(y_test)),
                    y=y_predictions,
                    color='steelblue',
                    label = 'predictions',
                    ax=ax)
    ax.legend()
    plt.show()

# Load data
df_train, df_test, df_test_RUL = load_data()

# Preprocess data. Remove unnecessary and highly correlated features, calculate RUL, and remove outliers
df_train, df_test = preprocess_data(df_train, df_test)

# Perform feature selection using backward stepwise regression
selected_features = select_features(df_train)

# Prepare train and test sets. Use selected features for train and test input
X_train = df_train[selected_features]
y_train = df_train.iloc[:,-1]

# Update test set to match RUL provided. Select last cycle for each engine in test set
df_test = df_test.groupby('engine').last().reset_index()

X_test = df_test[selected_features]
y_test = df_test_RUL

# Scale data using MinMaxScaler to improve performance
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train and evaluate model. Using GradientBoostingRegressor as the predictive model.
model = train_model(X_train, y_train)

# Evaluate the model by checking its score and errors
evaluate_model(model, X_test, y_test.values.ravel())

# Plot actual vs predicted Remaining Useful Life (RUL)
plot_actual_vs_prediction(y_test.values.ravel(), model.predict(X_test))