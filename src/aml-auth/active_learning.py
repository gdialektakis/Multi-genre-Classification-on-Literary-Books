from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from active_learning_methods import query_by_committee, ranked_batch_mode_sampling, random_active_learning, full_data_estimator, information_density
from data_processing import get_fully_processed, get_selected_genres


def prepare_data():
    books_df, genres_to_predict = get_fully_processed(genres_list=get_selected_genres(), multilabel=False)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(books_df['book_description_processed'])
    y = books_df['major_genre'].values

    return X, y, genres_to_predict


if __name__ == "__main__":
    X, y, genres_to_predict = prepare_data()

    n_samples_for_initial = 100
    n_queries = 10
    n_comittee_members = 3
    estimator = LogisticRegression(n_jobs=-1, max_iter=1000)

    print("------------------------------- \n Estimator using the whole data as labeled\n ")
    full_data_estimator.run(X, y, estimator)

    print("------------------------------- \n Query by committee\n ")
    num_of_annotated_samples = query_by_committee.run(X, y, n_samples_for_initial, n_queries, n_comittee_members, estimator)
    print("Number of annotations needed: {}".format(num_of_annotated_samples))

    print("------------------------------- \n Ranked batch mode sampling\n")
    batch_size = 3
    num_of_annotated_samples = ranked_batch_mode_sampling.run(X, y, n_samples_for_initial, n_queries, batch_size, estimator)
    print("Number of annotations needed: {}".format(num_of_annotated_samples))

    print("------------------------------- \n Random active learning \n ")
    num_of_annotated_samples = random_active_learning.run(X, y, n_samples_for_initial, n_queries, estimator)
    print("Number of annotations needed: {}".format(num_of_annotated_samples))

    print("------------------------------- \n Information density active learning \n ")
    num_of_annotated_samples = information_density.run(X, y, n_samples_for_initial, n_queries, estimator)
    print("Number of annotations needed: {}".format(num_of_annotated_samples))
