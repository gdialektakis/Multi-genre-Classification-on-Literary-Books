from collections import Counter
from imblearn.over_sampling import SMOTE
from data_processing import get_major_genre_split
from evaluation import evaluate_model
from sklearn.linear_model import LogisticRegression


def classifier(X, y):
    # model to use
    model = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300)
    model.fit(X, y)
    return model


def run():
    # TODO: ValueError: Expected n_neighbors <= n_samples,  but n_samples = 1, n_neighbors = 2
    # https://stackoverflow.com/questions/49395939/smote-initialisation-expects-n-neighbors-n-samples-but-n-samples-n-neighbo

    X_train, X_test, y_train, y_test = get_major_genre_split()

    # resample all classes but the majority class
    sm = SMOTE(sampling_strategy='not majority', random_state=42)

    X_res, y_res = sm.fit_resample(X_train, y_train)

    print('Original dataset shape %s' % Counter(y_train))
    print('Resampled dataset shape %s' % Counter(y_res))

    clf = classifier(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Model performance before SMOTE")
    evaluate_model(actual=y_test, predicted=y_pred)

    clf = classifier(X_res, y_res)
    y_pred = clf.predict(X_test)
    print("Model performance after SMOTE")
    evaluate_model(actual=y_test, predicted=y_pred)


if __name__ == "__main__":
    run()
