from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from evaluation import evaluate_model
from class_imbalance_methods.helper_functions import get_baseline_split, get_fully_processed_books_df
from class_imbalance_methods import smote, adasyn, easy_ensemble, text_augmentation


def run_classifiers(X_train, X_test, y_train, y_test):
    for classifier_name in ["logreg"]:#, "bayes", "rand_forest", "xgb"]:
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

        print("==============================")
        print('Results for {:}'.format(classifier))
        evaluate_model(y_test, y_pred, print_results=True)
        print("==============================")


def main():
    for representation in ["bow"]:#, "tf-idf"]:
        print("\n")
        print("++++++++")
        print(f"{representation}")
        print("++++++++")

        # Run baseline
        X_train, X_test, y_train, y_test = get_baseline_split(representation=representation)
        run_classifiers(X_train, X_test, y_train, y_test)

        # Text augmentation Experiments
        X_train, X_test, y_train, y_test = text_augmentation.run(books_df=get_fully_processed_books_df(), representation=representation)
        run_classifiers(X_train, X_test, y_train, y_test)

        # SMOTE Experiments
        X_smote, y_smote = smote.run(X_train, y_train)
        run_classifiers(X_smote, X_test, y_smote, y_test)

        # ADASYN Experiments
        X_adasyn, y_adasyn = adasyn.run(X_train, y_train)
        run_classifiers(X_adasyn, X_test, y_adasyn, y_test)

        # Easy Ensemble Experiments
        y_test, y_pred = easy_ensemble.run(X_train, X_test, y_train, y_test)
        evaluate_model(y_test, y_pred, print_results=True, average='weighted')


if __name__ == "__main__":
    main()
