import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取CSV文件，把第一列解析为日期
data = pd.read_csv("../data/us_map.csv", parse_dates=[0], index_col=0)

# 提取第4行数据
row_data = data.iloc[3]

# 将数据转换为每周数据
weekly_data = row_data.resample('W').sum()

# 绘制曲线图
plt.figure(figsize=(10, 5))
plt.plot(weekly_data.index, weekly_data.values)

plt.xlabel("Time")
plt.ylabel("Number of People")
plt.title("Weekly Data for the State: {}".format(row_data.name))
plt.xticks(rotation=45)

# 横轴尽可能少的显示时间点
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())  # 设置主要刻度为每月
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())  # 设置次要刻度为每周
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 设置刻度标签格式

# 保存和显示图形
plt.tight_layout()
plt.savefig("state_weekly_data.png")
plt.show()
