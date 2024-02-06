import matplotlib.dates as mdates
import pandas as pd
from matplotlib import rcParams, pyplot as plt
from matplotlib.ticker import MaxNLocator
from pandas.plotting import register_matplotlib_converters

config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

range_start = 120
range_num = 240


def draw_cpu():
    register_matplotlib_converters()

    predict_data = pd.read_csv('../datasets/predict_pa/cpu_total.csv')
    data = pd.read_csv('../datasets/hpa/cpu_total.csv')
    # Converting 'time' to datetime for plotting
    data['time'] = pd.to_datetime(data['time'])
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['value'] = predict_data['value'] * 100
    data['value'] = data['value'] * 100

    range_start = 180
    range_num = 300

    # Plotting
    plt.figure(figsize=(12, 9))
    plt.plot([i for i in range(0, 120)], predict_data['value'][0:120], label='实验组-预测式')
    plt.plot([i for i in range(0, 120)], data['value'][range_start:range_num], label='对照组-被动式')
    # plt.plot(data['value'], label='API')
    plt.ylabel('CPU使用率(%)')
    plt.title('数据服务API的CPU总使用率')
    plt.grid(True)
    plt.xticks()
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.legend()
    plt.savefig('./cpu_load.svg', format='svg', dpi=400,
                bbox_inches='tight')
    plt.show()


def draw_mem():
    register_matplotlib_converters()

    predict_data = pd.read_csv('../datasets/predict_pa/mem_total.csv')
    data = pd.read_csv('../datasets/hpa/mem_total.csv')
    # Converting 'time' to datetime for plotting
    data['time'] = pd.to_datetime(data['time'])
    data['value'] = data['value'] / (1024 ** 2)
    predict_data['time'] = pd.to_datetime(predict_data['time'])
    predict_data['value'] = predict_data['value'] / (1024 ** 2)

    range_start = 180
    range_num = 300

    # Plotting
    plt.figure(figsize=(12, 9))
    plt.plot([i for i in range(0, 120)], predict_data['value'][0:120], label='实验组-预测式')
    plt.plot([i for i in range(0, 120)], data['value'][range_start:range_num], label='对照组-被动式')
    # plt.plot(data['value'], label='API')
    plt.ylabel('内存占用量(MiB)')
    plt.title('数据服务API的内存总占用量')
    plt.grid(True)
    plt.xticks()
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.legend()
    plt.savefig('./mem_load.svg', format='svg', dpi=400,
                bbox_inches='tight')
    plt.show()


def draw_qps():
    register_matplotlib_converters()

    file_path = '../datasets/hpa/QPS.csv'
    data = pd.read_csv(file_path)
    # Converting 'time' to datetime for plotting
    data['time'] = pd.to_datetime(data['time'])

    range_start = 180
    range_num = 300
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot([i for i in range(0, 120)], data['value'][range_start:range_num], label='API')
    # plt.plot(data['value'], label='API')
    plt.ylabel('QPS')
    plt.title('数据服务API QPS')
    plt.grid(True)
    plt.xticks()
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.legend()
    plt.savefig('./load.svg', format='svg', dpi=400,
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
    plt.figure(figsize=(12, 9))
    plt.plot([i for i in range(0, 120)], predict_data['value'][0:120], label='实验组—预测式')
    plt.plot([i for i in range(0, 120)], reactive_data['value'][range_start:range_num],
             label='对照组—被动式')
    plt.ylabel('响应时间(ms)')
    plt.title('响应时间')
    plt.xticks()
    plt.yticks()
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.legend()
    plt.savefig('./response_time.svg', format='svg', dpi=400,
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

    plt.figure(figsize=(12, 9))
    plt.plot([i for i in range(0, 120)], predict_data['value'][0:120], label='实验组—预测式')
    plt.plot([i for i in range(0, 120)], reactive_data['value'][range_start:range_num],
             label='对照组—被动式')
    plt.ylabel('实例数')
    plt.title('API实例数变化')
    plt.grid(True)
    plt.xticks()
    plt.yticks()

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.legend()
    plt.savefig('./replicas.svg', format='svg', dpi=400,
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
    # draw_qps()
    # draw_cpu()
    # draw_mem()
    draw_request_time()
    # draw_replicas()
    # calc_metrics()
