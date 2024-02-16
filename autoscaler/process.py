import pandas as pd
from sklearn.impute import KNNImputer


def concat():
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


def fill_na():
    replica_df = pd.read_csv('../datasets/replica/replica_data.csv')

    # Initialize the KNN Imputer
    imputer = KNNImputer(n_neighbors=5)

    # Since KNNImputer works with numerical data, we'll exclude non-numerical columns if any
    numerical_data = replica_df.select_dtypes(include=['int64', 'float64'])

    # Fit the imputer and transform the data
    imputed_data = imputer.fit_transform(numerical_data)

    # Convert the imputed data back to a DataFrame
    imputed_data_df = pd.DataFrame(imputed_data, columns=numerical_data.columns)

    # Check if there are any missing values left
    print(imputed_data_df.isnull().sum(), imputed_data_df.head())

    columns_to_zero = ['CPU_AVG_USAGE', 'MEM_AVG_USAGE', 'CPU_TOTAL_USAGE', 'MEM_TOTAL_USAGE']
    imputed_data_df.loc[imputed_data_df['POD_COUNT'] == 0, columns_to_zero] = 0
    # Add the 'time' column back to the imputed and updated DataFrame
    imputed_data_df['time'] = replica_df['time']

    # Reorder the DataFrame to have 'time' as the first column
    imputed_data_df = imputed_data_df[['time'] + [col for col in imputed_data_df.columns if col != 'time']]

    # Verify the changes
    print(imputed_data_df.head(), (imputed_data_df[imputed_data_df['POD_COUNT'] == 0][columns_to_zero] == 0).all())

    imputed_data_df.to_csv('../datasets/replica/replica_data.csv', index=False)


def add_expected_value():
    replica_df = pd.read_csv('../datasets/replica/replica_data.csv')
    # 新增expected_CPU列，值全部赋值为1
    replica_df['expected_CPU_AVG_USAGE'] = 60.00
    # 新增expected_MEM列，值全部赋值为1
    replica_df['expected_MEM_AVG_USAGE'] = 70.00
    replica_df.to_csv('../datasets/replica/replica_data.csv', index=False)

    predict_df = pd.read_csv('../datasets/replica/predict_data.csv')
    predict_df['expected_CPU_AVG_USAGE'] = 60.00
    predict_df['expected_MEM_AVG_USAGE'] = 70.00
    predict_df.to_csv('../datasets/replica/predict_data.csv', index=False)


if __name__ == '__main__':
    # concat()
    # 勿动
    # concat_predict()
    # fill_na()
    add_expected_value()
