import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# project imports
from multi_label_classification import classifier_chains, one_vs_rest, rakel
from evaluation import evaluate_model, evaluate_per_label
from data_processing import get_processed_split, get_selected_genres


def estimators_to_try():
    estimators = []

    estimators.append(MultinomialNB(alpha=0.1))
    estimators.append(LogisticRegression(max_iter=1000, n_jobs=-1))
    estimators.append(RandomForestClassifier(criterion='entropy', n_estimators=200, n_jobs=-1))

    return estimators


def evaluate_and_store_results(y_test, y_pred, model, example_based_results, label_based_results, print_results=True):
    # example based evaluation
    accuracy_score, precision_score, \
    recall_score, f1_score, hamming_loss, classfication_report = evaluate_model(y_test, y_pred, print_results=print_results)

    # label based evaluation
    accuracy_per_label, precision_per_label, \
        recall_per_label, f1_per_label = evaluate_per_label(y_test, y_pred, print_results=True)

    example_based_results = example_based_results.append({'representation': representation, 
                                                          'estimator': str(estimator), 
                                                          'model': model,
                                                          'accuracy': accuracy_score, 
                                                          'precision': precision_score, 
                                                          'recall': recall_score,
                                                          'f1': f1_score, 
                                                          'hamming_loss': hamming_loss}, ignore_index=True)

    label_based_results = label_based_results.append({'representation': representation, 
                                                      'estimator': str(estimator), 
                                                      'model': model,
                                                      'accuracy': accuracy_per_label, 
                                                      'precision':precision_per_label, 
                                                      'recall':recall_per_label, 
                                                      'f1': f1_per_label}, ignore_index=True)

    return example_based_results, label_based_results


if __name__ == "__main__":
    # init dataframes with results
    example_based_results = pd.DataFrame(columns=['representation', 'model', 'estimator', 'accuracy', 'precision', 'recall', 'f1', 'hamming_loss'])
    label_based_results = pd.DataFrame(columns=['representation', 'model', 'estimator',  'accuracy', 'precision', 'recall', 'f1'])

    for representation in ["bow", "tf-idf"]:
        train_test_set = get_processed_split(genres_list=get_selected_genres(), vectorized=representation, multilabel=True)

        for estimator in estimators_to_try():
            print('Estimator: {}'.format(estimator))

            y_test, y_pred = classifier_chains.run(estimator, train_test_set)
            example_based_results, label_based_results = evaluate_and_store_results(y_test, 
                                                                                    y_pred, 
                                                                                    'classifier_chains', 
                                                                                    example_based_results, 
                                                                                    label_based_results, 
                                                                                    print_results=True)
            
            y_test, y_pred = one_vs_rest.run(estimator, train_test_set)
            example_based_results, label_based_results = evaluate_and_store_results(y_test, 
                                                                                    y_pred, 
                                                                                    'one_vs_rest', 
                                                                                    example_based_results, 
                                                                                    label_based_results, 
                                                                                    print_results=True)

            y_test, y_pred = rakel.run(estimator, train_test_set)
            example_based_results, label_based_results = evaluate_and_store_results(y_test, 
                                                                                    y_pred.toarray(), 
                                                                                    'rakel', 
                                                                                    example_based_results, 
                                                                                    label_based_results, 
                                                                                    print_results=True)
      
    # export results in csv
    example_based_results.to_csv('example_based_results.csv', index=False)
    label_based_results.to_csv('label_based_results.csv', index=False)