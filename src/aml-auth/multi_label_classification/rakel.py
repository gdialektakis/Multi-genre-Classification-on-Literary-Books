from skmultilearn.ensemble import RakelD


def run(classifier, train_test_set):
    X_train, X_test, y_train, y_test = train_test_set

    # init model and fit to train data
    rakel = RakelD(base_classifier=classifier)
    rakel.fit(X_train, y_train)

    # make predictions
    y_pred = rakel.predict(X_test)
    print('\n--------Rakel with {:}'.format(rakel))

    return y_test, y_pred
