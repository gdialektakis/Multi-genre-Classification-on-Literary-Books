from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from active_learning_methods import query_by_committee
from data_processing import get_fully_processed, get_selected_genres


def prepare_data():
    books_df, genres_to_predict = get_fully_processed(genres_list=get_selected_genres())

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform(books_df['book_description_processed'])
    y = books_df['major_genre'].values

    return X, y, genres_to_predict


if __name__ == "__main__":
    X, y, genres_to_predict = prepare_data()

    n_samples_for_intial = 1000    
    n_queries = 10
    n_comittee_members = 3
    estimator = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300)

    query_by_committee.run(X, y, n_samples_for_intial, n_queries, n_comittee_members, estimator)