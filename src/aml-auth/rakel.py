from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD

from data_processing import get_processed_split
from evaluation import evaluate_model


def run():
    X_train, X_test, y_train, y_test = get_processed_split()
    base_classifier = GaussianNB()

    rakel = RakelD(base_classifier=base_classifier)
    rakel.fit(X_train, y_train)
    y_pred = rakel.predict(X_test)

    evaluate_model(y_test, y_pred, print_results=True)


if __name__ == "__main__":
    run()
