from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

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
                     ('clf', RandomForestClassifier(n_estimators=100)),
                     ])

text_clf.fit(X_train, y_train)


y_pred = text_clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

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
y_score = text_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='RF ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
