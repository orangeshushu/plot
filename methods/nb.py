import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt

# 读取CSV文件
try:
    # class1 = pd.read_csv('positive_for_train.csv', encoding='utf-8')
    class1 = pd.read_csv('clean_positive.csv', encoding='utf-8')
except UnicodeDecodeError:
    class1 = pd.read_csv('clean_positive.csv', encoding='ISO-8859-1')

try:
    class2 = pd.read_csv('clean_negative.csv', encoding='utf-8')
except UnicodeDecodeError:
    class2 = pd.read_csv('clean_negative.csv', encoding='ISO-8859-1')

class1['text'].fillna('', inplace=True)
class2['label'].fillna('', inplace=True)

# 分配标签
class1['label'] = 1
class2['label'] = 0

# 合并数据
data = pd.concat([class1, class2])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
print(X_train.values)


# 特征提取
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train.values.astype('U'))
X_test_vec = vectorizer.transform(X_test)

# 训练朴素贝叶斯模型
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

sentence = "Guys I tested positive for covid so if you were at my concert yesterday please run some tests and stay at home ♡ https://t.co/FYXfK1o2oZ"
sentence_vec = vectorizer.transform([sentence])
res = nb.predict(sentence_vec)
print("~~~~~~~~")
print(res)
# 预测
y_pred = nb.predict(X_test_vec)
print("========")
print(y_pred)
y_pred_proba = nb.predict_proba(X_test_vec)[:, 1]
print(y_pred_proba)
print("========")


# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Accuracy: {accuracy:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')
print(classification_report(y_test, y_pred))

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
