import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 读取csv文件
df = pd.read_csv('figures/predict_real_cure.csv')

# 提取第二列和第三列的数据
x = df.iloc[:, 1].values
y = df.iloc[:, 2].values

# 对x和y分别进行平稳性验证
result_x = adfuller(x)
result_y = adfuller(y)

# 输出结果
print("x的平稳性验证结果：")
print("ADF Statistic: %f" % result_x[0])
print("p-value: %f" % result_x[1])
print("Critical Values:")
for key, value in result_x[4].items():
    print("\t%s: %.3f" % (key, value))

print("\ny的平稳性验证结果：")
print("ADF Statistic: %f" % result_y[0])
print("p-value: %f" % result_y[1])
print("Critical Values:")
for key, value in result_y[4].items():
    print("\t%s: %.3f" % (key, value))
