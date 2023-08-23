# Predicting Remaining Useful Life (RUL) Algorithm

## Author 
Isaac Monroy

## Project Description
This algorithm aims to predict the Remaining Useful Life (RUL) of engines based on sensor data. By using machine learning techniques to generate a model that can estimate the time before a given engine requires maintenance or replacement. The process includes preprocessing steps such as removing constant sensor readings, dropping highly correlated features, calculating the RUL, and removing outliers from the training data. It uses backward stepwise regression for feature selection and the Gradient Boosting Regressor for the training of the model. The model's performance is then evaluated using metrics like R-squared error, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Squared Log Error (MSLE). A plot of the actual RUL values against the predicted RUL values is also provided for visual representation of the model's performance.

## Libraries Used
- **NumPy**: Numerical computations
- **Pandas**: Dataframe manipulation
- **Statsmodels**: Statistical modeling
- **Scipy**: Z-score for outlier detection
- **Sklearn**: Machine learning models and preprocessing
- **Matplotlib & Seaborn**: Visualization

## How to Run
1. Ensure the required libraries are installed
2. Add the path to the CMaps dataset on the code
3. Run the script in your preferred IDE

## Input and Output
### Input:
- Training, testing, and RUL data from specified paths.

### Output:
- Predicted Remaining Useful Life (RUL) for engines
- Evaluation metrics such as R-squared error, RMSE, MAE, MSLE
- Plot of actual vs. predicted RUL values
