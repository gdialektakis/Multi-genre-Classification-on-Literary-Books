from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier


def run(X_train, X_test, y_train, y_test):
    print("######################")
    print("Easy Ensemble")
    print("######################")
    print("\n")

    print('Original dataset shape %s' % Counter(y_train))

    # resample all classes but the majority class
    eec = EasyEnsembleClassifier(sampling_strategy='not majority', replacement=True, random_state=42, n_jobs=-1)
    eec.fit(X_train, y_train)
    y_pred = eec.predict(X_test)
    y_proba = eec.predict_proba(X_test)

    return y_test, y_pred, y_proba


if __name__ == "__main__":
    run()
