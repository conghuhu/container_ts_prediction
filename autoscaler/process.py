import pandas as pd


def concat():
    cpu_df = pd.read_csv('../datasets/hpa/cpu_avg.csv')
    mem_df = pd.read_csv('../datasets/hpa/mem_avg.csv')
    count_df = pd.read_csv('../datasets/hpa/pod_count.csv')

    cpu_df['value'] = cpu_df['value'] * 100
    count_df['value'] = count_df['value'].astype(int)

    # 首先，我们重命名每个 DataFrame 中的 'value' 列，以便在合并后区分它们
    cpu_df = cpu_df.rename(columns={'value': 'CPU_AVG_USAGE'}).drop('metric', axis=1)
    mem_df = mem_df.rename(columns={'value': 'MEM_AVG_USAGE'}).drop('metric', axis=1)
    count_df = count_df.rename(columns={'value': 'POD_COUNT'}).drop('metric', axis=1)

    # 接着，我们按照 'time' 列合并这三个 DataFrame
    # 使用 'outer' 合并策略来确保所有时间点都被包含，即使某些时间点在某些 DataFrame 中不存在数据
    df_merged = cpu_df.merge(mem_df, on='time', how='outer').merge(count_df, on='time', how='outer')
    df_merged['CPU_TOTAL_USAGE'] = df_merged['CPU_AVG_USAGE'] * df_merged['POD_COUNT']
    df_merged['MEM_TOTAL_USAGE'] = df_merged['MEM_AVG_USAGE'] * df_merged['POD_COUNT']

    new_order = ['time', 'CPU_AVG_USAGE', 'MEM_AVG_USAGE', 'CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE', 'POD_COUNT']
    df_merged = df_merged[new_order]

    df_merged.to_csv('../datasets/replica/replica_data.csv', index=False)


def concat_predict():
    cpu_df = pd.read_csv('../datasets/replica/cpu_avg.csv')
    mem_df = pd.read_csv('../datasets/replica/mem_avg.csv')
    count_df = pd.read_csv('../datasets/replica/pod_count.csv')

    cpu_df['value'] = cpu_df['value'] * 100
    count_df['value'] = count_df['value'].astype(int)

    # 首先，我们重命名每个 DataFrame 中的 'value' 列，以便在合并后区分它们
    cpu_df = cpu_df.rename(columns={'value': 'CPU_AVG_USAGE'}).drop('metric', axis=1)
    mem_df = mem_df.rename(columns={'value': 'MEM_AVG_USAGE'}).drop('metric', axis=1)
    count_df = count_df.rename(columns={'value': 'POD_COUNT'}).drop('metric', axis=1)

    # 接着，我们按照 'time' 列合并这三个 DataFrame
    # 使用 'outer' 合并策略来确保所有时间点都被包含，即使某些时间点在某些 DataFrame 中不存在数据
    df_merged = cpu_df.merge(mem_df, on='time', how='outer').merge(count_df, on='time', how='outer')

    df_merged['CPU_TOTAL_USAGE'] = df_merged['CPU_AVG_USAGE'] * df_merged['POD_COUNT']
    df_merged['MEM_TOTAL_USAGE'] = df_merged['MEM_AVG_USAGE'] * df_merged['POD_COUNT']
    new_order = ['time', 'CPU_AVG_USAGE', 'MEM_AVG_USAGE', 'CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE', 'POD_COUNT']
    df_merged = df_merged[new_order]

    df_merged.to_csv('../datasets/replica/replica_predict_data.csv', index=False)


if __name__ == '__main__':
    concat()
    concat_predict()
