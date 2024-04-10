import os
import time

import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)


def process_data():
    # Load the datasets
    replica_data_path = '../datasets/hpa_week/replica_data.csv'
    replica_data = pd.read_csv(replica_data_path)
    # 描述数据集
    print("数据集描述如下：")
    print(replica_data.describe())

    # 添加时间特征
    replica_data['time'] = pd.to_datetime(replica_data['time'])
    replica_data['hour'] = replica_data['time'].dt.hour
    replica_data['day_of_week'] = replica_data['time'].dt.dayofweek
    replica_data.sort_values(by=['time'], inplace=True)

    # Preprocessing: Drop 'time' column and correct understanding of TOTAL_USAGE columns
    # replica_data_preprocessed = replica_data.drop(columns=['time'])

    # Preparing the features and target variable
    X = replica_data[['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE', 'hour', 'day_of_week', 'expected_CPU_AVG_USAGE',
                      'expected_MEM_AVG_USAGE']]
    # 添加交互特征
    X_poly = poly.fit_transform(X[['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE']])  # 只对CPU和内存使用率添加交互特征

    # 合并新的交互特征和原始特征
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE']))
    X_poly_df.drop('CPU_TOTAL_USAGE', axis=1, inplace=True)
    X_poly_df.drop('MEM_TOTAL_USAGE', axis=1, inplace=True)
    X = pd.concat([X.reset_index(drop=True), X_poly_df], axis=1)
    y = replica_data['POD_COUNT']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type='randomForest', train_mode='grid'):
    start_time = time.time()
    best_params = None
    # Training the Random Forest Regressor
    if model_type == 'randomForest':
        if train_mode == 'grid':
            # 定义要搜索的参数网格
            # Best parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}
            param_grid = {
                'n_estimators': [300, 500, 1000],  # 树的数量
                'max_depth': [None, 10, 20, 30],  # 树的最大深度
                'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最少样本数
                'min_samples_leaf': [1, 2, 4],  # 在叶节点上所需的最小样本数
            }
            rd = RandomForestRegressor(random_state=42)
            # 设置GridSearchCV
            grid_search = GridSearchCV(estimator=rd, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error',
                                       n_jobs=-1, verbose=2)

            # 进行搜索
            grid_search.fit(X_train, y_train)

            # 输出最佳参数
            print("Best parameters:", grid_search.best_params_)
            print("Best score:", -grid_search.best_score_)

            # 使用最佳参数的模型进行预测
            rf_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            rf_model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                             random_state=42)
            rf_model.fit(X_train, y_train)
    elif model_type == 'linearRegression':
        rf_model = LinearRegression()
        rf_model.fit(X_train, y_train)
    elif model_type == 'elasticNet':
        if train_mode == 'grid':
            # 定义要搜索的参数网格
            # Best parameters found:  {'alpha': 1.0, 'l1_ratio': 0.9}
            param_grid = {
                'alpha': [0.1, 1.0, 10.0],  # 正则化强度
                'l1_ratio': [0.1, 0.5, 0.9]  # L1比例在L1和L2正则化之间的混合参数
            }
            en = ElasticNet(random_state=42)
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=en, param_grid=param_grid, cv=3,
                                       scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)

            # Best parameters and best score
            print("Best parameters found: ", grid_search.best_params_)
            print("Best score found: ", grid_search.best_score_)
            # Use the best parameters to re-train the model
            rf_model = grid_search.best_estimator_

            best_params = grid_search.best_params_
        else:
            rf_model = ElasticNet(alpha=0.1, l1_ratio=0.1, random_state=42)
            rf_model.fit(X_train, y_train)
    elif model_type == 'xgboost':
        if train_mode == 'grid':
            # Best parameters found:  {'colsample_bytree': 1, 'early_stopping_rounds': 10, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2, 'n_estimators': 400, 'reg_alpha': 1, 'reg_lambda': 0, 'subsample': 1}
            param_grid = {
                'n_estimators': [100, 300, 500, 1000],
                'max_depth': [i for i in range(3, 11, 2)],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.5, 0.7, 1],
                'colsample_bytree': [0.5, 0.7, 1],
                'early_stopping_rounds': [10],
                'min_child_weight': [i for i in range(0, 11, 2)],
                'gamma': [i for i in range(0, 11, 2)],
                'reg_alpha': [0, 0.5, 1],
                'reg_lambda': [0, 0.5, 1],
            }
            # Initialize the XGBRegressor
            xgb_regressor = XGBRegressor(random_state=42)
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=3,
                                       scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

            # Fit GridSearchCV
            grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

            # Best parameters and best score
            print("Best parameters found: ", grid_search.best_params_)
            print("Best score found: ", grid_search.best_score_)
            # Use the best parameters to re-train the model
            rf_model = grid_search.best_estimator_

            best_params = grid_search.best_params_
        else:
            rf_model = XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.1, subsample=1,
                                    colsample_bytree=1, min_child_weight=0,
                                    early_stopping_rounds=10, random_state=42, gamma=0, reg_alpha=1, reg_lambda=0.5)
            rf_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    elif model_type == 'lightgbm':
        if train_mode == 'grid':
            # Best parameters found:  {'colsample_bytree': 0.75, 'early_stopping_round': 10, 'learning_rate': 0.05, 'max_depth': -1, 'n_estimators': 300, 'num_leaves': 60, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 0.5}
            param_grid = {
                'num_leaves': [28, 31, 60, 80],  # Adjust based on dataset size and complexity
                'max_depth': [3, 5, 10, 15, -1],  # -1 means no limit
                'learning_rate': [0.1, 0.01, 0.05],
                'n_estimators': [100, 200, 300, 400],
                'early_stopping_round': [10],
                'subsample': [0.5, 0.7, 1],
                'colsample_bytree': [0.5, 0.75, 1],
                'reg_alpha': [0, 1e-2, 0.5, 1, 1e2, 1e3],
                'reg_lambda': [0, 1e-2, 0.5, 1, 1e3, 1e3],
            }
            lgbm = LGBMRegressor(random_state=42)
            grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
                                       verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train,
                            eval_set=[(X_test, y_test)])
            # Best parameters and best score
            print("Best parameters found: ", grid_search.best_params_)
            print("Best score found: ", grid_search.best_score_)

            # Use the best parameters to re-train the model
            rf_model = grid_search.best_estimator_

            best_params = grid_search.best_params_
        else:
            rf_model = LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.1, subsample=0.5, num_leaves=60,
                                     colsample_bytree=1, reg_alpha=1, reg_lambda=1,
                                     early_stopping_round=10, random_state=42)
            rf_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    else:
        raise Exception(f"Model type {model_type} not supported")
    end_time = time.time()  # 记录结束时间
    print(f"模型训练耗时: {end_time - start_time} 秒")

    f = open("result_reg.txt", 'a')
    f.write(model_type + "  \n")
    f.write('{} Best Params: {}'.format(model_type, best_params))
    f.write('\n')
    f.close()
    return rf_model


def test_model(rf_model, X_test, y_test, model_type='randomForest'):
    # Predicting on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_type} Model Evaluation:\nMSE: {mse}\nMAE: {mae}\nRMSE: {rmse}\nR2 Score: {r2}")
    f = open("result_reg.txt", 'a')
    f.write('MSE: {}, MAE: {}, RMSE: {}, R2: {}'.format(mse, mae, rmse, r2))
    f.write('\n')
    f.write('\n')
    f.close()


def predict(rf_model):
    predict_data_path = '../datasets/hpa_week/predict/predict_data.csv'
    predict_data = pd.read_csv(predict_data_path)
    print("预测集描述如下：")
    print(predict_data.describe())

    # 使用训练好的模型对预测数据集进行预测
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['hour'] = predict_data['time'].dt.hour
    predict_data['day_of_week'] = predict_data['time'].dt.dayofweek
    predict_data.sort_values(by=['time'], inplace=True)

    predict_X = predict_data[['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE', 'hour', 'day_of_week', 'expected_CPU_AVG_USAGE',
                              'expected_MEM_AVG_USAGE']]
    predict_X_poly = poly.transform(predict_X[['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE']])
    predict_X_poly_df = pd.DataFrame(predict_X_poly,
                                     columns=poly.get_feature_names_out(['CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE']))
    predict_X_poly_df.drop('CPU_TOTAL_USAGE', axis=1, inplace=True)
    predict_X_poly_df.drop('MEM_TOTAL_USAGE', axis=1, inplace=True)
    predict_X = pd.concat([predict_X.reset_index(drop=True), predict_X_poly_df], axis=1)

    start_time = time.time()
    # Predicting the expected POD_COUNT using the trained model
    predicted_pod_count = rf_model.predict(predict_X)
    end_time = time.time()
    print(f"Predicting the expected POD_COUNTs using the trained model took {end_time - start_time} seconds")

    # Rounding the predicted POD_COUNTs to the nearest integer
    rounded_predicted_pod_count = np.round(predicted_pod_count).astype(int)

    # Displaying the rounded predicted POD_COUNTs
    print("Predicted POD_COUNTs (ceil to the largest integer):")
    print(rounded_predicted_pod_count)


def get_current_memory_gb() -> float:
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024.


def count_info(func):
    def float_info():
        pid = os.getpid()
        p = psutil.Process(pid)
        info_start = p.memory_full_info().uss / 1024
        func()
        info_end = p.memory_full_info().uss / 1024
        print("程序占用了内存" + str(info_end - info_start) + "KB")

    return float_info


@count_info
def main_process():
    nums = 100
    for i in range(nums):
        print(f"第{i}次训练")
        model = train_model(X_train, y_train, model_type, 'use')
        test_model(model, X_test, y_test, model_type)
        predict(model)


if __name__ == '__main__':
    model_type = 'xgboost'
    X_train, X_test, y_train, y_test = process_data()
    main_process()

    # model = train_model(X_train, y_train, model_type, 'grid')
