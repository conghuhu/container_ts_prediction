def quick_sort(arr, start, end):
    """
    arr: 数组
    start: 开始
    end: 结束
    """
    # 如果开始下标 >= 结束下标
    if start >= end:
        # 直接返回
        return

    partition_idx = partition(arr, start, end)
    # 对左和右分别递归进行快速排序
    quick_sort(arr, start, partition_idx - 1)
    quick_sort(arr, partition_idx + 1, end)

# 寻找中心轴
def partition(arr, start_index, end_index):
    # 基准值
    pivot = arr[start_index]
    mark = start_index
    for i in range(start_index + 1, end_index + 1):
        if arr[i] < pivot:
            mark += 1
            swap(arr, i, mark)
    swap(arr, start_index, mark)
    return mark

# 交换函数
def swap(arr, x, y):
    if x == y:
        return
    arr[x], arr[y] = arr[y], arr[x]

def main():
    # 1. 初始化一个数组
    arr = [6, 5, 8, 1, 3, 7, 9, 5, 2]
    # 2. 调用quick_sort函数
    quick_sort(arr, 0, len(arr) - 1)
    print(arr)


if __name__ == "__main__":
    # 调用main函数
    main()
