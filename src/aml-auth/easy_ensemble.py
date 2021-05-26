from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
from data_processing import get_major_genre_split
from evaluation import evaluate_model

def run():
    X_train, X_test, y_train, y_test = get_major_genre_split()

    print('Original dataset shape %s' % Counter(y_train))

    # resample all classes but the majority class
    eec = EasyEnsembleClassifier(sampling_strategy='not majority', random_state=42)
    eec.fit(X_train, y_train)
    y_pred = eec.predict(X_test)

    evaluate_model(actual=y_test, predicted=y_pred, print_results=True, average='weighted')


if __name__ == "__main__":
    run()