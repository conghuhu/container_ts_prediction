import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor


def process_data():
    # Load the datasets
    replica_data_path = '../datasets/replica/replica_data.csv'
    replica_data = pd.read_csv(replica_data_path)
    # 添加时间特征
    replica_data['time'] = pd.to_datetime(replica_data['time'])
    replica_data['hour'] = replica_data['time'].dt.hour
    replica_data['day_of_week'] = replica_data['time'].dt.dayofweek

    # Preprocessing: Drop 'time' column and correct understanding of TOTAL_USAGE columns
    # replica_data_preprocessed = replica_data.drop(columns=['time'])

    # Preparing the features and target variable
    X = replica_data[['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE', 'hour', 'day_of_week']]
    y = replica_data['POD_COUNT']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type='randomForest'):
    # Training the Random Forest Regressor
    if model_type == 'randomForest':
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
    elif model_type == 'linearRegression':
        rf_model = LinearRegression()
        rf_model.fit(X_train, y_train)
    elif model_type == 'xgboost':
        param_grid = {
            'n_estimators': [100, 200, 500, 1000, 1500],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.5, 0.75, 1],
            'colsample_bytree': [0.5, 0.75, 1],
            'early_stopping_rounds': [10]
        }
        # Initialize the XGBRegressor
        xgb_regressor = XGBRegressor(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=3,
                                   scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

        # Best parameters and best score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score found: ", grid_search.best_score_)
        # Use the best parameters to re-train the model
        rf_model = grid_search.best_estimator_
    elif model_type == 'elasticNet':
        rf_model = ElasticNetCV(cv=5, random_state=42)
        rf_model.fit(X_train, y_train)
    elif model_type == 'lightgbm':
        param_grid = {
            'num_leaves': [31, 50, 70],  # Adjust based on dataset size and complexity
            'max_depth': [10, 20, -1],  # -1 means no limit
            'learning_rate': [0.1, 0.01, 0.05],
            'n_estimators': [100, 200, 500, 1000, 1500],
            'early_stopping_round': [10]
        }
        lgbm = LGBMRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                   verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)])
        # Best parameters and best score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score found: ", grid_search.best_score_)

        # Use the best parameters to re-train the model
        rf_model = grid_search.best_estimator_
    else:
        raise Exception(f"Model type {model_type} not supported")
    return rf_model


def test_model(rf_model, X_test, y_test):
    # Predicting on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:\nMSE: {mse}\nMAE: {mae}\nRMSE: {rmse}\nR2 Score: {r2}")


def predict(rf_model):
    predict_data_path = '../datasets/replica/predict_data.csv'
    predict_data = pd.read_csv(predict_data_path)
    # 使用训练好的模型对预测数据集进行预测
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['hour'] = predict_data['time'].dt.hour
    predict_data['day_of_week'] = predict_data['time'].dt.dayofweek
    predict_X = predict_data[['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE', 'hour', 'day_of_week']]

    # Predicting the expected POD_COUNT using the trained model
    predicted_pod_count = rf_model.predict(predict_X)

    # Rounding the predicted POD_COUNTs to the nearest integer
    rounded_predicted_pod_count = np.round(predicted_pod_count).astype(int)

    # Displaying the rounded predicted POD_COUNTs
    print("Predicted POD_COUNTs (ceil to the largest integer):")
    print(rounded_predicted_pod_count)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = process_data()
    model = train_model(X_train, y_train, 'xgboost')
    test_model(model, X_test, y_test)
    predict(model)
