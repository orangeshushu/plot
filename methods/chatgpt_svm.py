import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score

# 加载数据
def load_data():
    tweets = []
    labels = []

    with open("chatGPT_gerator_data/recover.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # 跳过标题行
        for row in reader:
            tweets.append(row[0])
            labels.append(1)

    with open("chatGPT_gerator_data/other.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # 跳过标题行
        for row in reader:
            tweets.append(row[0])
            labels.append(0)

    return tweets, labels

# 主函数
def main():
    tweets, labels = load_data()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tweets)
    y = np.array(labels)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练SVM分类器
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # 评估
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy: ", accuracy_score(y_test, y_pred))

    # 画ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()

    # 对给定文本进行预测
    given_text = """Just because we're social distancing, doesn't mean we can't have some fun! Here are some ideas:

-Get creative in the kitchen and try out some new recipes
-Have a virtual movie night with friends
-Start a new project- like learning a new language or crafting
-Go for a walk or run in nature- enjoy the fresh air!"""
    X_given = vectorizer.transform([given_text])
    pred = clf.predict(X_given)

    print("Prediction for the given text:")
    print("Recover" if pred[0] == 1 else "Other")

if __name__ == "__main__":
    main()
