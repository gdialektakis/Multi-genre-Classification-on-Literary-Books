from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from data_processing import get_processed_split, label_encode_genres, get_fully_processed
from evaluation import evaluate_model
from imblearn.over_sampling import SMOTE, ADASYN


def run_classifiers(X_train, X_test, y_train, y_test):

    for classifier_name in ["logreg", "bayes", "rand_forest", "xgb"]:
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
        print(f"Results for {classifier_name}")
        evaluate_model(y_test, y_pred, print_results=True)
        print("==============================")


def get_baseline_split(representation=None):
    genres_list = ['Romance', 'Adventure', 'Young Adult', 'Space', 'Historical', 'Adult',
                   'Speculative Fiction', 'War', 'Apocalyptic']

    if representation == "bow":
        X_train, X_test, y_train, y_test = get_processed_split(genres_list=genres_list, vectorized="bow")
    elif representation == "tf-idf":
        X_train, X_test, y_train, y_test = get_processed_split(genres_list=genres_list, vectorized="tf-idf")
    else:
        raise ValueError("Invalid representation")

    y_train = label_encode_genres(y_train, genres_list)
    y_test = label_encode_genres(y_test, genres_list)

    y_train = y_train['genre_label'].to_frame()
    y_test = y_test['genre_label'].to_frame()

    return X_train, X_test, y_train, y_test


def main():
    for representation in ["bow", "tf-idf"]:
        print("\n")
        print("++++++++")
        print(f"{representation}")
        print("++++++++")

        X_train, X_test, y_train, y_test = get_baseline_split(representation=representation)
        run_classifiers(X_train, X_test, y_train, y_test)

        # SMOTE Experiments
        smote_sampler = SMOTE(sampling_strategy='not majority', n_jobs=-1)
        X_smote, y_smote = smote_sampler.fit_resample(X_train, y_train)
        print("######################")
        print("SMOTE")
        print("######################")
        print("\n")
        run_classifiers(X_smote, X_test, y_smote, y_test)

        # ADASYN Experiments
        adasyn_sampler = ADASYN(sampling_strategy='not majority', n_jobs=-1)
        X_adasyn, y_adasyn = adasyn_sampler.fit_resample(X_train, y_train)
        print("######################")
        print("ADASYN")
        print("######################")
        print("\n")
        run_classifiers(X_adasyn, X_test, y_adasyn, y_test)


if __name__ == "__main__":
    main()
