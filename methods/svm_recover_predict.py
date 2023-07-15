import pickle

def predict_new(text):
    with open('svm_positive.pkl', 'rb') as f:
        svm, vectorizer = pickle.load(f)
    X_new = vectorizer.transform([text])
    prob = svm.predict_proba(X_new)
    return svm.predict(X_new)[0], prob[0][1]

predict_text = "He tests positive for covid today"
prediction_res, prob = predict_new(predict_text)
# 1: recover, 0 : not
print("Prediction result is: ", prediction_res)
print("Reover prob is ：", prob)
