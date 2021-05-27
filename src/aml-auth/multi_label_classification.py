from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# project imports
from multi_label_classification import classifier_chains, one_vs_rest, rakel
from evaluation import evaluate_model
from data_processing import get_processed_split, get_selected_genres


def estimators_to_try():
    estimators = []
    
    estimators.append(LogisticRegression(solver='lbfgs', random_state=0, max_iter=300))
    estimators.append(MultinomialNB())

    return estimators


if __name__ == "__main__":
    train_test_set = get_processed_split(genres_list=get_selected_genres())

    for estimator in estimators_to_try():
        print('Estimator: {}'.format(estimator))

        y_test, y_pred = classifier_chains.run(estimator, train_test_set)
        evaluate_model(y_test, y_pred, print_results=True)

        y_test, y_pred = one_vs_rest.run(estimator, train_test_set)
        evaluate_model(y_test, y_pred, print_results=True)

        #y_test, y_pred = rakel.run(estimator, train_test_set)
        #evaluate_model(y_test, y_pred, print_results=True)