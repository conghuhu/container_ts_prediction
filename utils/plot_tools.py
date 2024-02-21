import os
import time

from matplotlib import pyplot as plt, rcParams

config = {
    "font.family": "serif",
    "font.size": 20,
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"],
    "font.weight": "normal",
}
rcParams.update(config)

# 定义字体font1
font1 = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": 20,
}
font2 = {
    # 'family': prop.get_name(),
    "family": "SimSun",
    "size": 20,
    "weight": "normal",
}


def plot_loss_data(train_loss, vali_loss, test_loss, setting, run_type):
    # 使用Matplotlib绘制线图
    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, marker="o", label="train_loss")
    plt.plot(vali_loss, marker="o", label="vali_loss")
    plt.plot(test_loss, marker="o", label="test_loss")

    # 添加标题
    plt.title("loss results")
    # 显示图例
    plt.legend()

    folder_path = "./loss_imgs/" + setting + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.tight_layout()
    plt.savefig(folder_path + "/loss.svg", format="svg", dpi=400, bbox_inches="tight")

    if run_type == "ide":
        plt.show()


def closePlots():
    plt.clf()
    plt.cla()
    plt.close("all")
    time.sleep(0.5)
