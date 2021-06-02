from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from active_learning_methods import query_by_committee, ranked_batch_mode_sampling, random_active_learning, full_data_estimator, information_density
from data_processing import get_fully_processed, get_selected_genres
from sklearn.naive_bayes import MultinomialNB


def prepare_data():
    books_df, genres_to_predict = get_fully_processed(genres_list=get_selected_genres(), multilabel=False)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=7000)
    X = vectorizer.fit_transform(books_df['book_description_processed'])
    y = books_df['major_genre'].values

    return X, y, genres_to_predict


if __name__ == "__main__":
    X, y, genres_to_predict = prepare_data()

    n_samples_for_initial = 100
    n_queries = 10
    n_comittee_members = 3

    print("Initial samples for training: {}".format(n_samples_for_initial))

    for classifier_name in ["bayes"]:  #, "logreg", "bayes"
        if classifier_name == "logreg":
            classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
        elif classifier_name == "bayes":
            classifier = MultinomialNB(alpha=0.01)
        elif classifier_name == "rand_forest":
            classifier = RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=-1)
        else:
            raise ValueError(f"Unknown classifier {classifier_name}")

        # print("------------------------------- \n Estimator {} using the whole data as labeled\n ".format(classifier_name))
        # full_data_estimator.run(X, y, classifier)

        # print("------------------------------- \n Query by committee\n ")
        # num_of_annotated_samples = query_by_committee.run(X, y, n_samples_for_initial, n_queries, n_comittee_members, classifier)
        # print("Number of annotations needed for {} : {}".format(classifier_name, num_of_annotated_samples))

        # print("------------------------------- \n Ranked batch mode sampling\n")
        # batch_size = 5
        # num_of_annotated_samples = ranked_batch_mode_sampling.run(X, y, n_samples_for_initial, n_queries, batch_size, classifier)
        # print("Number of annotations needed for {} : {}".format(classifier_name, num_of_annotated_samples))
        #
        print("------------------------------- \n Information density active learning \n ")
        num_of_annotated_samples = information_density.run(X, y, n_samples_for_initial, n_queries, classifier)
        print("Number of annotations needed for {} : {}".format(classifier_name, num_of_annotated_samples))
        #
        # print("------------------------------- \n Random active learning \n ")
        # num_of_annotated_samples = random_active_learning.run(X, y, n_samples_for_initial, n_queries, classifier)
        # print("Number of annotations needed for {} : {}".format(classifier_name, num_of_annotated_samples))


