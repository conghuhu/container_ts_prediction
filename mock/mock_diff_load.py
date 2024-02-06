import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

# 生成示例数据
np.random.seed(0)  # 确保数据的可重复性
time_points = np.arange(0, 20)  # 时刻从0到19
# 服务A、B、C在0-10时刻的CPU使用率变化趋势相同，之后呈现不同的趋势
cpu_usage_A = np.concatenate([np.sin(time_points[:11] * 0.5) + np.random.normal(0, 0.1, 11),
                              np.sin(time_points[11:] * 0.5) * 0.9 + np.random.normal(0, 0.1, 9) + 1])
cpu_usage_B = np.concatenate([np.sin(time_points[:11] * 0.5) + np.random.normal(0, 0.1, 11),
                              np.sin(time_points[11:] * 0.5) * -0.8 + np.random.normal(0, 0.1, 9)])
cpu_usage_C = np.concatenate([np.sin(time_points[:11] * 0.5) + np.random.normal(0, 0.1, 11),
                              np.sin(time_points[11:] * 0.5) * 1.2 + np.random.normal(0, 0.1, 9) + 2])
# 计算原始数据的最大最小值，以便正确缩放
min_cpu_usage = min(cpu_usage_A.min(), cpu_usage_B.min(), cpu_usage_C.min())
max_cpu_usage = max(cpu_usage_A.max(), cpu_usage_B.max(), cpu_usage_C.max())

# 调整范围到0-99
scale_factor = 99 / (max_cpu_usage - min_cpu_usage)
cpu_usage_A = (cpu_usage_A - min_cpu_usage) * scale_factor
cpu_usage_B = (cpu_usage_B - min_cpu_usage) * scale_factor
cpu_usage_C = (cpu_usage_C - min_cpu_usage) * scale_factor

cpu_usage_A = [28.58384327, 36.58896922, 46.38788347, 52.97425565, 50.04484871, 36.15265625,
               29.97756005, 15.92810722, 6.56553367, 2.61365965, 2.42581214, 36.37893953,
               43.71088272, 43.60533941, 42.63750384, 44.28234867, 43.06893374, 46.09480446,
               45.18943596, 44.23185186]
cpu_usage_B = [18.50954678, 37.18044947, 46.12114758, 46.01295858, 50.98341915, 35.03931899,
               27.86720597, 15.84450177, 10.38331653, 5.08439411, 2.45125749, 38.52136827,
               27.61186143, 26.82877266, 24.39013146, 23.32067371, 21.86803403, 20.36625262,
               19.86956809, 17.16476234]
cpu_usage_C = [22.02031696, 32.34138872, 40.12211922, 52.29723359, 44.49736523, 37.41094758,
               24.83688653, 18.09567598, 3.04019675, 1.15902949, 0., 52.28484989,
               62.12286392, 74.40838885, 89.47149527, 98.406132, 99., 94.20559897,
               88.19987801, 88.18847047]

# 创建DataFrame
df = pd.DataFrame({
    'Time': time_points,
    'CPU_Usage_A': cpu_usage_A,
    'CPU_Usage_B': cpu_usage_B,
    'CPU_Usage_C': cpu_usage_C
})

print(cpu_usage_A)
print(cpu_usage_B)
print(cpu_usage_C)

config = {
    "font.family": 'serif',
    "font.size": 16,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['CPU_Usage_A'], label='API-A')
plt.plot(df['Time'], df['CPU_Usage_B'], label='API-B')
plt.plot(df['Time'], df['CPU_Usage_C'], label='API-C')
plt.axvline(11, color='blue', linestyle='--', linewidth=2)
plt.xlabel('时间步')
plt.ylabel('CPU使用率(%)')
plt.title('不同数据服务API的CPU使用率')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('./diff_cpu_load.svg', format='svg', dpi=400,
            bbox_inches='tight')
plt.show()
