import time

from matplotlib import pyplot as plt


def plot_loss_data(data, loss_name):
    # 使用Matplotlib绘制线图
    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o', label="{}".format(loss_name))

    # 添加标题
    plt.title("loss results Plot")

    # 显示图例
    plt.legend()

    plt.show()


def closePlots():
    plt.clf()
    plt.cla()
    plt.close("all")
    time.sleep(0.5)
