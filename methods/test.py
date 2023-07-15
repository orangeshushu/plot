import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)
error = 0.1 * np.random.randn(100) # 生成误差数据
print(error)
# 计算误差区间的上下限
lower_bound = y - error
upper_bound = y + error
# print(lower_bound, upper_bound)
# 绘制曲线和误差区间
plt.plot(x, y, color='blue', label='Curve')
plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.5, label='Error')

# 设置标题和标签
plt.title('Curve with Error Bounds')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# 显示图形
plt.show()
