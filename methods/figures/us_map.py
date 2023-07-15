import pandas as pd

# 读取原始文件
df = pd.read_csv('../data/us_map.csv')

# 对每一行的第二个元素开始的缺失值进行后向填充
df.iloc[:, 1:] = df.iloc[:, 1:].fillna(method='backfill', axis=1)

# 将处理后的数据保存到新文件中
df.to_csv('new.csv', index=False)
