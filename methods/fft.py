import csv
import numpy as np
from scipy.signal import hilbert

# 从CSV文件中读取数据
with open('figures/predict_real_cure.csv') as csvfile:
    data = list(csv.reader(csvfile))

# 将数据转换为NumPy数组
data_array = np.array(data[1:])
print(data_array)
# 获取第二列和第三列的数据
signal_1 = data_array[:,1]
signal_2 = data_array[:,2]

# 计算每个信号的希尔伯特变换
analytic_signal_1 = hilbert(signal_1)
analytic_signal_2 = hilbert(signal_2)

# 计算每个信号的瞬时相位
phase_1 = np.unwrap(np.angle(analytic_signal_1))
phase_2 = np.unwrap(np.angle(analytic_signal_2))

# 计算相位差
phase_diff = phase_2 - phase_1

# 打印相位差的均值
print("相位差的均值为:", np.max(phase_diff))
