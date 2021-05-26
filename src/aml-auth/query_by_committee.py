import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from modAL.models import ActiveLearner, Committee
from modAL.disagreement import max_disagreement_sampling

from data_processing import get_fully_processed
from ranked_batch_mode_sampling import delete_rows_csr


def prepare_data():
    books_df, genres_to_predict = get_fully_processed()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform(books_df['book_description_processed'])
    y = books_df['major_genre'].values

    return X, y


def create_random_pool_and_initial_sets(X, y, n_samples_for_intial):
    training_indices = np.random.choice(range(X.shape[0]), size=n_samples_for_intial, replace=False)

    X_train = X[training_indices]
    y_train = y[training_indices]

    X_pool = delete_rows_csr(X, training_indices)
    y_pool = np.delete(y, training_indices)

    return X_train, y_train, X_pool, y_pool


def run():
    X, y = prepare_data()

    # start timer
    start_time = time.time()

    # model to use
    model = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300)

    n_comittee_members = 3

    # init list of different learners 
    learners = []

    for member_idx in range(n_comittee_members):
        X_train, y_train, X_pool, y_pool = create_random_pool_and_initial_sets(X, y, 100)

        learners.append(ActiveLearner(estimator=model,
                        X_training=X_train, y_training=y_train))
        
    # init committee
    committee = Committee(learner_list=learners, query_strategy=max_disagreement_sampling)

    # unqueried_score = committee.score(X_pool, y_pool)
    # print('Score over unqueried samples'.format(unqueried_score))

    performance_history = []
    n_queries = 30

    for query in range(n_queries):

        # get sample from pool
        query_idx, query_instance = committee.query(X_pool)

        # retrain comittee with new sample
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )

        # save accuracy score
        performance_history.append(committee.score(X, y))

        # remove queried instance from pool
        X_pool = delete_rows_csr(X_pool, query_idx)
        y_pool = np.delete(y_pool, query_idx)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    print(performance_history)


if __name__ == "__main__":
    run()