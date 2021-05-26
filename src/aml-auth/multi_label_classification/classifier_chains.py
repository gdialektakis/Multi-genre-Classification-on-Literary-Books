from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain


def run(classifier, train_test_set):
    X_train, X_test, y_train, y_test = train_test_set

    # init model and fit to train data
    chain = ClassifierChain(classifier, order='random', random_state=0)
    chain.fit(X_train, y_train)

    # make predictions
    y_pred = chain.predict(X_test)
    print('\n--------Classifier chains with {:}'.format(classifier))

    return y_test, y_pred
