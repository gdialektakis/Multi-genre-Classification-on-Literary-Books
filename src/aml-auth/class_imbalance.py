import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from evaluation import evaluate_model, imbalanced_evaluate
from class_imbalance_methods.helper_functions import get_baseline_split, get_fully_processed_books_df
from class_imbalance_methods import smote, adasyn, easy_ensemble, text_augmentation


def run_classifiers(X_train, X_test, y_train, y_test, representation, technique_name):
    results_label_f1 = "{}-{}-f1".format(representation, technique_name)
    results_label_auc = "{}-{}-auc".format(representation, technique_name)
    classifier_results = {
        results_label_f1: [],
        results_label_auc: []
    }

    for classifier_name in ["logreg", "bayes", "rand_forest"]:
        if classifier_name == "logreg":
            classifier = LogisticRegression(max_iter=10000, n_jobs=-1)
        elif classifier_name == "bayes":
            classifier = MultinomialNB(alpha=0.01)
        elif classifier_name == "rand_forest":
            classifier = RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=-1)
        elif classifier_name == "xgb":
            classifier = XGBClassifier(
                scale_pos_weight=100,
                eta=0.01,
                max_depth=10,
                min_child_weight=5,
                objective="binary:logistic",
                eval_metric="auc"
            )
        else:
            raise ValueError(f"Unknown classifier {classifier_name}")

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

        print("==============================")
        print('Results for {:}'.format(classifier))
        f1_score, auc = imbalanced_evaluate(y_test, y_pred, y_proba, print_results=True)
        print("==============================")

        # results_label_f1 = "{}-{}-{}-{}".format(representation, technique_name, classifier_name, "f1")
        # results_label_auc = "{}-{}-{}-{}".format(representation, technique_name, classifier_name, "auc")

        classifier_results[results_label_f1].append(f1_score)
        classifier_results[results_label_auc].append(auc)

    return classifier_results


def main():
    results = {}

    for representation in ["bow", "tf-idf"]:
        print("\n")
        print("++++++++")
        print(f"{representation}")
        print("++++++++")

        # # Run baseline
        X_train, X_test, y_train, y_test = get_baseline_split(representation=representation)
        experiment_metrics = run_classifiers(X_train, X_test, y_train, y_test, representation, "baseline")
        results.update(experiment_metrics)

        # SMOTE Experiments
        X_smote, y_smote = smote.run(X_train, y_train)
        experiment_metrics = run_classifiers(X_smote, X_test, y_smote, y_test, representation, "SMOTE")
        results.update(experiment_metrics)

        # ADASYN Experiments
        X_adasyn, y_adasyn = adasyn.run(X_train, y_train)
        experiment_metrics = run_classifiers(X_adasyn, X_test, y_adasyn, y_test, representation, "ADASYN")
        results.update(experiment_metrics)

        # Easy Ensemble Experiments
        y_test, y_pred, y_proba = easy_ensemble.run(X_train, X_test, y_train, y_test)
        f1_score, auc = imbalanced_evaluate(y_test, y_pred, y_proba, print_results=True, average='micro')
        results_label_f1 = "{}-{}-{}".format(representation, "easy_ensemble", "f1")
        results_label_auc = "{}-{}-{}".format(representation, "easy_ensemble", "auc")
        results.update({
            results_label_f1: [f1_score, f1_score, f1_score],
            results_label_auc: [auc, auc, auc]
        })

        # Text augmentation Experiments
        X_train_aug, X_test_aug, y_train_aug, y_test_aug = text_augmentation.run(books_df=get_fully_processed_books_df(),
                                                                                 representation=representation)
        experiment_metrics = run_classifiers(X_train_aug, X_test_aug, y_train_aug, y_test_aug, representation,
                                             "text_augmentation")
        results.update(experiment_metrics)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv("class_imbalance_experiments.csv")


if __name__ == "__main__":
    main()
