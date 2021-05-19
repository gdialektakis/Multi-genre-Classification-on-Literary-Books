from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB

from data_processing import text_conditioning, get_n_most_frequent_genres, filter_out_genres, genres_to_onehot, get_processed_split
from evaluation import evaluate_model



def run(classifier):
    X_train, X_test, y_train, y_test = get_processed_split()
    chain = ClassifierChain(classifier, order='random', random_state=0)
    chain.fit(X_train, y_train)
    # chain.predict_proba(X_test)
    y_pred = chain.predict(X_test)
    print('\n--------Classifier chains with {:}'.format(classifier))
    evaluate_model(y_test, y_pred, print_results=True)
    return y_test, y_pred


if __name__ == "__main__":
    # Logistic Regression
    logreg = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300)
    run(logreg)
    # Naive Bayes
    MNB = MultinomialNB()
    run(MNB)
