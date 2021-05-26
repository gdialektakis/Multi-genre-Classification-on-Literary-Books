from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from data_processing import get_processed_split
from evaluation import evaluate_model


def run(model):
    """Run OneVsRest classifier given an estimator
    """

    # get splitted dataset where X is the tfidf representation of books and y the corresponding labels
    X_train, X_test, y_train, y_test = get_processed_split()
    
    # init model and fit to train data
    model = OneVsRestClassifier(model)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    print('\n--------OneVsRest with {:}'.format(model.estimator))
    evaluate_model(y_test, y_pred, print_results=True)

    return y_test, y_pred


if __name__ == "__main__":
    # Logistic Regression
    logreg = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300)
    run(logreg)

    # Naive Bayes
    MNB = MultinomialNB()
    run(MNB)