import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pickle

try:
    data1 = pd.read_csv("chatGPT_gerator_data/other.csv", encoding='utf-8')
except UnicodeDecodeError:
    data1 = pd.read_csv("chatGPT_gerator_data/other.csv", encoding='ISO-8859-1')
try:
    data2 = pd.read_csv("chatGPT_gerator_data/recover.csv", encoding='utf-8')
except UnicodeDecodeError:
    data2 = pd.read_csv("chatGPT_gerator_data/recover.csv", encoding='ISO-8859-1')

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
sentence = "Just because I have COVID-19 doesn't mean I'm giving up. I'm still fighting and I'm not going to let this virus beat me."
sentence_vec = vectorizer.transform([sentence])
res = svm.predict(sentence_vec)
prob = svm.predict_proba(sentence_vec)
print("~~~~~~~~")
print(res)
print(prob)
print("~~~~~~~~")

y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# 保存模型
with open('svm_chatgpt_recover.pkl', 'wb') as f:
    pickle.dump((svm, vectorizer), f)

# 预测新数据
def predict_new(text):
    with open('svm_chatgpt_recover.pkl', 'rb') as f:
        clf, vectorizer = pickle.load(f)
    X_new = vectorizer.transform([text])
    prob = clf.predict_proba(X_new)
    return clf.predict(X_new)[0],prob[0][1]

predict_text = "I have recovered from COVID today"
prediction_res, prob = predict_new(predict_text)
print("Prediction result is: ", prediction_res)
print("Reover prob is ：", prob)
