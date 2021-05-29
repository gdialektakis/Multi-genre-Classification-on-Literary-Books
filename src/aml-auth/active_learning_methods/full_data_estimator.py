from sklearn import metrics
from sklearn.model_selection import train_test_split


def run(X, y, estimator):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average='micro')
    print(f"Accuracy score: {accuracy:.4f}")
    print(f"F1 score: {f1_score:.4f}")
    return
