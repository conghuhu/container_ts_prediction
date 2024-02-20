import pandas as pd
from matplotlib import rcParams, pyplot as plt, font_manager
from matplotlib.ticker import MaxNLocator

config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
    "font.weight": "normal"
}
rcParams.update(config)

print("+++++++++++++++++++++")

font_path = 'C:\\Users\\15841\\AppData\\Local\\Microsoft\\Windows\\Fonts\\宋体-粗体.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
# 定义字体font1
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
font2 = {
    # 'family': prop.get_name(),
    'family': 'SimSun',
    'size': 20,
    'weight': 'normal',
}

# plt.style.use(['science', 'no-latex', 'cjk-sc-font'])

range_start = 120
range_num = 240


def draw_pod_resource():
    mem_df = pd.read_csv('../datasets/hpa/cpu_total.csv')
    pod_df = pd.read_csv('../datasets/hpa/pod_count.csv')

    range_start = 180
    range_num = 300
    mem_df = mem_df[range_start:range_num]
    pod_df = pod_df[range_start - 2:range_num - 2]
    # 二者对value归一化
    mem_df['value'] = mem_df['value'] / mem_df['value'].max()
    pod_df['value'] = pod_df['value'] / pod_df['value'].max()

    plt.figure(figsize=(12, 8))
    plt.plot([i for i in range(0, range_num - range_start)], mem_df['value'], label='实例所需资源')
    plt.plot([i for i in range(0, range_num - range_start)], pod_df['value'], label='实例扩容资源')
    # plt.plot(data['value'], label='API')
    # plt.ylabel('CPU使用率(%)', font2)
    plt.xlabel('时间步', font2)
    plt.grid(True)
    plt.xticks()
    plt.tight_layout()
    plt.legend(prop=font2)
    plt.savefig('./pod_resource.svg', format='svg', dpi=800,
                bbox_inches='tight')
    plt.show()


def draw_cpu():
    # register_matplotlib_converters()

    predict_data = pd.read_csv('../datasets/predict_pa/cpu_total.csv')
    hpa_data = pd.read_csv('../datasets/hpa/cpu_total.csv')
    # Converting 'time' to datetime for plotting
    hpa_data['time'] = pd.to_datetime(hpa_data['time'])
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['value'] = predict_data['value'] * 100
    hpa_data['value'] = hpa_data['value'] * 100

    range_start = 180
    range_num = 300
    predict_data = predict_data[0:120]
    hpa_data = hpa_data[range_start:range_num]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot([i for i in range(0, range_num - range_start)], predict_data['value'], label='实验组-预测式')
    plt.plot([i for i in range(0, range_num - range_start)], hpa_data['value'], label='对照组-被动式')
    # plt.plot(data['value'], label='API')
    plt.ylabel('CPU使用率(%)', font2)
    # plt.title('数据服务API的CPU总使用率')
    plt.grid(True)
    plt.xticks()
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.legend(prop=font2)
    plt.savefig('./cpu_load.svg', format='svg', dpi=800,
                bbox_inches='tight')
    plt.show()


def draw_mem():
    # register_matplotlib_converters()

    predict_data = pd.read_csv('../datasets/predict_pa/mem_total.csv')
    hpa_data = pd.read_csv('../datasets/hpa/mem_total.csv')
    # Converting 'time' to datetime for plotting
    hpa_data['time'] = pd.to_datetime(hpa_data['time'])
    hpa_data['value'] = hpa_data['value'] / (1024 ** 2)
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['value'] = predict_data['value'] / (1024 ** 2)

    range_start = 180
    range_num = 300
    predict_data = predict_data[0:120]
    hpa_data = hpa_data[range_start:range_num]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot([i for i in range(0, range_num - range_start)], predict_data['value'], label='实验组-预测式')
    plt.plot([i for i in range(0, range_num - range_start)], hpa_data['value'], label='对照组-被动式')
    # plt.plot(data['value'], label='API')
    plt.ylabel('内存占用量(MiB)', font2)
    # plt.title('数据服务API的内存总占用量')
    plt.grid(True)
    plt.xticks()
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.legend(prop=font2)
    plt.savefig('./mem_load.svg', format='svg', dpi=800,
                bbox_inches='tight')
    plt.show()


def draw_qps():
    # register_matplotlib_converters()

    file_path = '../datasets/hpa/QPS.csv'
    data = pd.read_csv(file_path)
    # Converting 'time' to datetime for plotting
    data['time'] = pd.to_datetime(data['time'])

    range_start = 0
    range_num = 1321
    # range_start = 180
    # range_num = 300
    data = data[range_start:range_num]
    data['value'] = data['value'] * 2

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot([i for i in range(0, range_num - range_start)], data['value'], label='API')
    # plt.plot(data['value'], label='API')
    plt.ylabel('QPS')
    # plt.title('数据服务API QPS')
    plt.grid(True)
    plt.xticks()
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    # plt.legend()
    plt.legend(loc="upper right", prop=font1)
    plt.savefig('./load.svg', format='svg', dpi=800,
                bbox_inches='tight')
    plt.show()


def draw_request_time():
    predict_data = pd.read_csv('../datasets/predict_pa/response_time.csv')
    reactive_data = pd.read_csv('../datasets/hpa/response_time.csv')
    # Converting 'time' to datetime for plotting
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['value'] = predict_data['value'] * 1000
    reactive_data['time'] = pd.to_datetime(reactive_data['time'])
    reactive_data['value'] = reactive_data['value'] * 1000

    range_start = 180
    range_num = 300
    # range_start = 120
    # range_num = 180
    plt.figure(figsize=(12, 8))
    plt.plot([i for i in range(0, 120)], predict_data['value'][0:120], label='实验组—预测式')
    plt.plot([i for i in range(0, 120)], reactive_data['value'][range_start:range_num],
             label='对照组—被动式')
    plt.ylabel('响应时间(ms)', font2)
    # plt.title('响应时间')
    plt.xticks()
    plt.yticks()
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.legend(prop=font2)
    plt.savefig('./response_time.svg', format='svg', dpi=800,
                bbox_inches='tight')
    plt.show()


def draw_replicas():
    predict_data = pd.read_csv('../datasets/predict_pa/pod_count.csv')
    reactive_data = pd.read_csv('../datasets/hpa/pod_count.csv')
    # Converting 'time' to datetime for plotting
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    reactive_data['time'] = pd.to_datetime(reactive_data['time'])
    range_start = 180
    range_num = 300

    plt.figure(figsize=(12, 8))
    plt.plot([i for i in range(0, 120)], predict_data['value'][0:120], label='实验组—预测式')
    plt.plot([i for i in range(0, 120)], reactive_data['value'][range_start:range_num],
             label='对照组—被动式')
    plt.ylabel('实例数', font2)
    # plt.title('API实例数变化')
    plt.grid(True)
    plt.xticks()
    plt.yticks()

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.legend(prop=font2)
    plt.savefig('./replicas.svg', format='svg', dpi=800,
                bbox_inches='tight')
    plt.show()


def calc_metrics():
    """
    计算实验metrics
    :return:
    """
    predict_data = pd.read_csv('../datasets/predict_pa/response_time.csv')
    reactive_data = pd.read_csv('../datasets/hpa/response_time.csv')
    # Converting 'time' to datetime for plotting
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['value'] = predict_data['value'] * 1000
    reactive_data['time'] = pd.to_datetime(reactive_data['time'])
    reactive_data['value'] = reactive_data['value'] * 1000

    range_start = 180
    range_num = 300

    print(predict_data['value'].max())
    print(predict_data['value'].mean())
    print(reactive_data['value'][range_start:range_num].mean())

    # print(reactive_data.iloc[range_start])
    # print(reactive_data.iloc[range_num])

    count1 = 0
    for index, item in predict_data.iterrows():
        if float(item['value']) > 200.00:
            count1 += 1
    print(count1)

    # 27个
    count = 0
    for index, item in reactive_data.iterrows():
        if float(item['value']) > 200.00:
            count += 1

    print(count)

    predict_cpu_avg_data = pd.read_csv('../datasets/predict_pa/cpu_avg.csv')
    reactive_cpu_avg_data = pd.read_csv('../datasets/hpa/cpu_avg.csv')
    print(predict_cpu_avg_data['value'].mean())
    print(reactive_cpu_avg_data['value'].mean())

    predict_mem_avg_data = pd.read_csv('../datasets/predict_pa/mem_avg.csv')
    reactive_mem_avg_data = pd.read_csv('../datasets/hpa/mem_avg.csv')
    print(predict_mem_avg_data['value'].mean())
    print(reactive_mem_avg_data['value'].mean())


if __name__ == '__main__':
    draw_qps()
    # draw_cpu()
    # draw_mem()
    # draw_request_time()
    # draw_replicas()
    # calc_metrics()
    # draw_pod_resource()
