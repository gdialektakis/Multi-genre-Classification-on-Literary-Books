from sklearn import metrics
from sklearn.model_selection import train_test_split


def run(X, y, estimator):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy:.4f}")
    return
