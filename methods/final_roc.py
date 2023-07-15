import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#LR
# 读取训练集

try:
    df_neg = pd.read_csv("negative_for_train.csv", encoding='utf-8')
except UnicodeDecodeError:
    df_neg = pd.read_csv("negative_for_train.csv", encoding='ISO-8859-1')
try:
    df_pos = pd.read_csv("positive_for_train.csv", encoding='utf-8')
except UnicodeDecodeError:
    df_pos = pd.read_csv("positive_for_train.csv", encoding='ISO-8859-1')

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

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("LR Classification Report:")
print(classification_report(y_test, y_pred))

# 绘制 ROC 和precision-recall曲线
y_score = model.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, y_score)
lr_precision, lr_recall, lr_thresholds = precision_recall_curve(y_test, y_score)
lr_auc_pr = metrics.average_precision_score(y_test, y_score)
lr_f1_scores = 2 * (lr_precision * lr_recall) / (lr_precision + lr_recall)
lr_best_threshold = lr_thresholds[np.argmax(lr_f1_scores)]
print("Best Threshold: ", lr_best_threshold)

lr_roc_auc = auc(lr_fpr, lr_tpr)
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve for Naive Bayes')
# plt.show()
print("============")

#SVM
try:
    data1 = pd.read_csv("negative_for_train.csv", encoding='utf-8')
except UnicodeDecodeError:
    data1 = pd.read_csv("negative_for_train.csv", encoding='ISO-8859-1')
try:
    data2 = pd.read_csv("positive_for_train.csv", encoding='utf-8')
except UnicodeDecodeError:
    data2 = pd.read_csv("positive_for_train.csv", encoding='ISO-8859-1')

print("okkokok")
data1['label'] = 0
data2['label'] = 1
data = pd.concat([data1, data2], ignore_index=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'].values.astype('U'))
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='rbf', probability=True)
print("training")
svm.fit(X_train, y_train)
print("over")
# sentence = "I test positive for covid today"
# sentence_vec = vectorizer.transform([sentence])
# res = svm.predict(sentence_vec)
# prob = svm.predict_proba(sentence_vec)

y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

svm_accuracy = accuracy_score(y_test, y_pred)
svm_precision = precision_score(y_test, y_pred)
svm_recall = recall_score(y_test, y_pred)
svm_f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1-score: {svm_f1:.4f}")

y_score = svm.predict_proba(X_test)[:, 1]
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, y_score)
svm_precision, svm_recall, svm_thresholds = precision_recall_curve(y_test, y_score)
svm_auc_pr = metrics.average_precision_score(y_test, y_score)
svm_f1_scores = 2 * (svm_precision * svm_recall) / (svm_precision + svm_recall)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_score)
svm_roc_auc = roc_auc_score(y_test, y_score)
print("fpr, svm_fpr", fpr, svm_fpr)
# fpr, tpr, _ = roc_curve(y_test, y_proba)
# svm_roc_auc = roc_auc_score(y_test, y_proba)

# NB
# 读取CSV文件
try:
    # class1 = pd.read_csv('positive_for_train.csv', encoding='utf-8')
    nb_class1 = pd.read_csv('clean_positive.csv', encoding='utf-8')
except UnicodeDecodeError:
    nb_class1 = pd.read_csv('clean_positive.csv', encoding='ISO-8859-1')

try:
    nb_class2 = pd.read_csv('clean_negative.csv', encoding='utf-8')
except UnicodeDecodeError:
    nb_class2 = pd.read_csv('clean_negative.csv', encoding='ISO-8859-1')

nb_class1['text'].fillna('', inplace=True)
nb_class2['label'].fillna('', inplace=True)

# 分配标签
nb_class1['label'] = 1
nb_class2['label'] = 0

# 合并数据
data = pd.concat([nb_class1, nb_class2])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# 特征提取
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train.values.astype('U'))
X_test_vec = vectorizer.transform(X_test)

# 训练朴素贝叶斯模型
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

y_pred = nb.predict(X_test_vec)
y_pred_proba = nb.predict_proba(X_test_vec)[:, 1]

# 评估模型性能
nb_accuracy = accuracy_score(y_test, y_pred)
nb_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Accuracy: {nb_accuracy:.2f}')
print(f'ROC AUC: {nb_roc_auc:.2f}')
print(classification_report(y_test, y_pred))

nb_accuracy = accuracy_score(y_test, y_pred)
nb_precision = precision_score(y_test, y_pred)
nb_recall = recall_score(y_test, y_pred)
nb_f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {nb_accuracy:.4f}")
print(f"Precision: {nb_precision:.4f}")
print(f"Recall: {nb_recall:.4f}")
print(f"F1-score: {nb_f1:.4f}")

y_score = nb.predict_proba(X_test_vec)[:, 1]
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, y_score)
nb_precision, nb_recall, nb_thresholds = precision_recall_curve(y_test, y_score)
nb_auc_pr = metrics.average_precision_score(y_test, y_score)
nb_f1_scores = 2 * (nb_precision * nb_recall) / (nb_precision + nb_recall)


# 绘制ROC曲线
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_test, y_pred_proba)

########################################################
# 训练KNN
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

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', KNeighborsClassifier()),
                     ])

text_clf.fit(X_train, y_train)

y_pred = text_clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

# 评估模型
print("KNN performance:")
knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall: {knn_recall:.4f}")
print(f"F1-score: {knn_f1:.4f}")

# 绘制 ROC 曲线
y_score = text_clf.predict_proba(X_test)[:, 1]
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, y_score)
knn_roc_auc = auc(knn_fpr, knn_tpr)

knn_precision, knn_recall, knn_thresholds = precision_recall_curve(y_test, y_score)
knn_auc_pr = metrics.average_precision_score(y_test, y_score)
knn_f1_scores = 2 * (knn_precision * knn_recall) / (knn_precision + knn_recall)

plt.figure()
plt.plot(lr_fpr, lr_tpr, label='LR ROC curve (area = %0.5f)' % lr_roc_auc)
plt.plot(svm_fpr, svm_tpr, label='SVM ROC curve (area = %0.5f)' % svm_roc_auc)
plt.plot(nb_fpr, nb_tpr, label='NB ROC curve (area = %0.5f)' % nb_roc_auc)
plt.plot(knn_fpr, knn_tpr, label='KNN ROC curve (area = %0.5f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('Receiver Operating Characteristic (ROC).jpg', dpi=300)
plt.show()


# plt.figure()
# plt.plot(lr_recall, lr_precision, label='LR (AP = %0.2f)' % lr_auc_pr)
# plt.plot(svm_recall, svm_precision, label='SVM (AP = %0.2f)' % svm_auc_pr)
# plt.plot(nb_recall, nb_precision, label='NB (AP = %0.2f)' % nb_auc_pr)
# plt.plot(knn_recall, knn_precision, label='KNN (AP = %0.2f)' % knn_auc_pr)
# print(lr_recall, lr_precision)
# print(svm_recall, svm_precision)
# print(nb_recall, nb_precision)
# print(knn_recall, knn_precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.savefig('precision_recall_curve.jpg', dpi=300)
# plt.legend(loc="lower left")
# plt.savefig('precision_recall_curve.jpg', dpi=300)
# plt.show()
