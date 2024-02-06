import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from pandas.plotting import register_matplotlib_converters

config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def draw_load():
    register_matplotlib_converters()

    file_path = '../datasets/hpa/cpu_total.csv'
    data = pd.read_csv(file_path)
    # Converting 'time' to datetime for plotting
    data['time'] = pd.to_datetime(data['time'])

    data['value'] = data['value'] * 100

    range_start = 45
    range_num = 105
    # range_num = 410

    # Plotting
    plt.figure(figsize=(12, 6))
    # plt.plot(data['time'][range_start:range_num], data['value_scaled'][range_start:range_num] * 3, label='API')
    plt.plot(data['time'], data['value'], label='API')
    plt.ylabel('CPU使用率(%)')
    plt.title('数据服务API的CPU总使用率')
    plt.grid(True)
    plt.xticks(rotation=30, fontsize=14)
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.legend()
    plt.savefig('./cpu_load.svg', format='svg', dpi=400,
                bbox_inches='tight')
    plt.show()

def draw_mem():
    register_matplotlib_converters()

    file_path = '../datasets/hpa/mem_total.csv'
    data = pd.read_csv(file_path)
    # Converting 'time' to datetime for plotting
    data['time'] = pd.to_datetime(data['time'])

    data['value'] = data['value'] / (1024 ** 2)

    range_start = 45
    range_num = 105
    # range_num = 410

    # Plotting
    plt.figure(figsize=(12, 6))
    # plt.plot(data['time'][range_start:range_num], data['value_scaled'][range_start:range_num] * 3, label='API')
    plt.plot(data['time'], data['value'], label='API')
    plt.ylabel('内存占用量(MiB)')
    plt.title('数据服务API的内存总占用量')
    plt.grid(True)
    plt.xticks(rotation=30, fontsize=14)
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
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

    range_start = 45
    range_num = 105
    # range_num = 410

    # Plotting
    plt.figure(figsize=(12, 6))
    # plt.plot(data['time'][range_start:range_num], data['value_scaled'][range_start:range_num] * 3, label='API')
    plt.plot(data['time'], data['value'], label='API')
    plt.ylabel('QPS')
    plt.title('数据服务API QPS')
    plt.grid(True)
    plt.xticks(rotation=30, fontsize=14)
    # plt.yticks(fontsize=20)

    # Improve formatting of time axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.legend()
    plt.savefig('./load.svg', format='svg', dpi=400,
                bbox_inches='tight')
    plt.show()


predict_replicas = [1, 1, 2, 2, 2, 4, 4, 4, 5, 5, 6, 6, 6, 6, 4, 4, 4, 3, 3, 3]
active_replicas = []


def stretch_series(series, target_length):
    factor = target_length // len(series)  # Stretching factor
    stretched_series = [item for item in series for _ in range(factor)]
    return stretched_series


def draw_replicas():
    predict_replicas = [1, 2, 2, 4, 4, 4, 5, 5, 8, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3, 3]
    active_replicas = [1, 1, 2, 2, 2, 3, 4, 4, 4, 6, 8, 8, 8, 8, 6, 6, 4, 4, 3, 3]
    # Extending the series
    predict_replicas = stretch_series(predict_replicas, 60)
    active_replicas = stretch_series(active_replicas, 60)

    plt.figure(figsize=(12, 6))
    plt.plot(predict_replicas, label='predict')
    plt.plot(active_replicas, label='active')
    plt.ylabel('replicas', fontsize=20)
    plt.title('Changes in the number of instances', fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig('./replicas.svg', format='svg', dpi=400,
                bbox_inches='tight')
    plt.show()


def draw_request_time():
    x = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]
    predict_replicas = [31, 47, 53, 49, 46, 57, 102, 123, 154, 153, 150, 146, 141, 123, 95, 87, 88, 86, 65, 58, 42]
    active_replicas = [31, 73, 53, 74, 63, 84, 130, 170, 196, 201, 195, 183, 131, 113, 85, 84, 81, 73, 63, 56, 45]
    print(len(x), len(predict_replicas), len(active_replicas))

    plt.figure(figsize=(12, 6))
    plt.plot(x, predict_replicas, label='predict')
    plt.plot(x, active_replicas, label='active')
    plt.ylabel('response time(ms)', fontsize=20)
    plt.title('Response time', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig('./request_time.svg', format='svg', dpi=400,
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_qps()
    draw_load()
    draw_mem()
    # draw_replicas()
    # draw_request_time()
