# Predictive Maintenance Model Compilation

## Author
Isaac Monroy

## Project Description
The objective of this algorithm is to offer versatility in regards to choosing a model for predicting a time series sensor dataset for predictive maintenance. The chosen models, LSTM, Random Forest Regressor, and XGBRegressor, have been tested with different time-series datasets and have provided the best prediction results.

## Libraries Used
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Keras**: For building, training, and evaluating the LSTM model.
- **Scikit-learn**: For preprocessing steps, model training, handling missing values, metrics for model evaluation, and hyperparameter tuning.
- **XGBoost**: For using the XGBoost model.

## How to Run
1. Load your dataset as a CSV file.
2. Separate features and labels, and split the data into training and testing sets.
3. Preprocess the training and testing data.
4. Choose the desired model or grid search option.
5. Run the selected model or perform grid search with the corresponding data setup.
6. If using LSTM, reshape the data to fit the model.
7. Follow the code comments for additional customization and options.

## Input and Output
- **Input**: Time series sensor dataset with target values.
- **Output**: The trained model, Mean Absolute Error for training and testing sets, and if applicable, the best parameters and mean absolute error from grid search.
