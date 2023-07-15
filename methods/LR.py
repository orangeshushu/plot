import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 读取训练集

try:
    df_neg = pd.read_csv("negative_for_train.csv", encoding='utf-8')
except UnicodeDecodeError:
    df_neg = pd.read_csv("negative_for_train.csv", encoding='ISO-8859-1')
try:
    df_pos = pd.read_csv("positive_for_train.csv", encoding='utf-8')
except UnicodeDecodeError:
    df_pos = pd.read_csv("positive_for_train.csv", encoding='ISO-8859-1')

print("okkokok")
df_neg['label'] = 0
df_pos['label'] = 1
data = pd.concat([df_neg, df_pos], ignore_index=True)

# 特征提取
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X = vectorizer.fit_transform(data['text'].values.astype('U'))
X = tfidf_transformer.fit_transform(X)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 训练模型
model = LogisticRegression()
print("traing")
model.fit(X_train, y_train)
print("over")

# 预测结果
y_pred = model.predict(X_test)

sentence = "I know he tested positive for covid"
sentence_vec = vectorizer.transform([sentence])
res = model.predict(sentence_vec)
print("~~~~~~~~")
print(res)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 绘制 ROC 曲线
y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
