## 1.数据背景

数据来自华为云数据湖探索（Data Lake Insight，简称DLI），它是一个Serverless的弹性大数据分析服务。对于用户来说，提交SQL/Spark/Flink
作业需要购买队列（Queue），并将作业指定到购买的队列中执行。队列（Queue）的概念可以认为是资源的容器，它在作业真实执行时是一个计算集群，队列存在不同的规格，
单位是CU（计算单元Compute Unit），1CU等于1核4GB，即16CU的队列代表着总资源16核64GB的计算集群。数据每5分钟会进行一次采集，
假设集群内节点间的任务调度平均，数据中的CPU_USAGE是集群中各节点平均值。

## 2.训练集

选取了43个队列的性能采集数据作为训练数据，每个队列之间相互独立。
`train.csv`为数据集。

## 3.测试集

训练和测试集按照0.8 0.2的比例进行划分。

## 4.字段说明：

| 字段                 | 类型     | 说明                                        |
|--------------------|--------|-------------------------------------------|
| QUEUE_ID           | INT    | 队列标识，每个ID代表一个唯一的队列                        |
| CU                 | INT    | 队列规格，不同规格的资源大小不一样。1CU为1核4GB。              |
| STATUS             | STRING | 队列状态，当前队列的状态是否可用                          |
| QUEUE_TYPE         | STRING | 队列类型，不同类型适用于不同的任务，常见的有通用队列（general）和SQL队列 |
| PLATFORM           | STRING | 队列平台，创建队列的机器平台                            |
| CPU_USAGE          | INT    | CPU使用率，集群中各机器节点的CPU平均使用率                  |
| MEM_USAGE          | INT    | 内存使用率，集群中各机器节点的内存平均使用率                    |
| LAUNCHING_JOB_NUMS | INT    | 提交中的作业数，即正在等待执行的作业                        |
| RUNNING_JOB_NUMS   | INT    | 运行中的作业数                                   |
| SUCCEED_JOB_NUMS   | INT    | 已完成的作业数                                   |
| CANCELLED_JOB_NUMS | INT    | 已取消的作业数                                   |
| FAILED_JOB_NUMS    | INT    | 已失败的作业数                                   |
| DOTTING_TIME       | BIGINT | 采集时间，每5分钟进行一次采集                           |
| RESOURCE_TYPE      | STRING | 资源类型，创建队列的机器类型                            |
| DISK_USAGE         | INT    | 磁盘使用                                      |
