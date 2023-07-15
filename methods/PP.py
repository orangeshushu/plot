import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 读取csv文件
df = pd.read_csv('figures/predict_real_cure.csv')

# 提取第二列和第三列的数据
x = df.iloc[:, 1].values
y = df.iloc[:, 2].values

# 对x和y分别进行PP检验
result_x = adfuller(x, autolag='AIC', regression='ct')
result_y = adfuller(y, autolag='AIC', regression='ct')

# 输出结果
print("x的PP检验结果：")
print("Test Statistic: %f" % result_x[0])
print("p-value: %f" % result_x[1])
print("Critical Values:")
for key, value in result_x[4].items():
    print("\t%s: %.3f" % (key, value))

print("\ny的PP检验结果：")
print("Test Statistic: %f" % result_y[0])
print("p-value: %f" % result_y[1])
print("Critical Values:")
for key, value in result_y[4].items():
    print("\t%s: %.3f" % (key, value))
