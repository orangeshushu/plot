import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x)

# 创建子图对象
fig, ax1 = plt.subplots()

# 在子图1中绘制曲线1
ax1.plot(x, y1, color='red', label='Sin(x)')

# 创建子图2并共享x轴
ax2 = ax1.twinx()

# 在子图2中绘制曲线2
ax2.plot(x, y2, color='blue', label='Exp(x)')

# 设置标题和标签
ax1.set_xlabel('X')
ax1.set_ylabel('Sin(x)')
ax2.set_ylabel('Exp(x)')

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# 显示图形
plt.show()
