import numpy as np


def MAE(pred: np.ndarray, true: np.ndarray):
    """
    平均绝对误差，L1范数损失
    :param pred:
    :param true:
    :return:
    """
    return np.mean(np.abs(pred - true))


def MSE(pred: np.ndarray, true: np.ndarray):
    """
    均方误差，又被称为 L2范数损失
    :param pred:
    :param true:
    :return:
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray):
    """
    均方根误差 RMSE
    :param pred:
    :param true:
    :return:
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred: np.ndarray, true: np.ndarray):
    """
    平均绝对百分比误差, 是MAE的加权版本
    :param pred:
    :param true:
    :return:
    """
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred: np.ndarray, true: np.ndarray):
    """
    均方百分比误差
    :param pred:
    :param true:
    :return:
    """
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def write_to_file(setting, mse, mae, rmse, mape, mspe):
    f = open("result_forecast.txt", 'a')
    f.write(setting + "  \n")
    f.write('mse: {}, mae: {}, rmse: {}, mape: {}, mspe: {}'.format(mse, mae, rmse, mape, mspe))
    f.write('\n')
    f.write('\n')
    f.close()
