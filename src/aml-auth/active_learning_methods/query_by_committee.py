import numpy as np
import time
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import max_disagreement_sampling
from active_learning_methods.helper_functions import delete_rows_csr
from sklearn import metrics


def create_random_pool_and_initial_sets(X, y, n_samples_for_intial):
    np.random.seed(0)
    training_indices = np.random.choice(range(X.shape[0]), size=n_samples_for_intial, replace=False)

    X_train = X[training_indices]
    y_train = y[training_indices]

    X_pool = delete_rows_csr(X, training_indices)
    y_pool = np.delete(y, training_indices)

    return X_train, y_train, X_pool, y_pool


def run(X, y, n_samples_for_intial, n_queries, n_comittee_members, estimator):
    # start timer
    start_time = time.time()

    # init list of different learners 
    learners = []

    X_train, y_train, X_pool, y_pool = create_random_pool_and_initial_sets(X, y, n_samples_for_intial)

    for member_idx in range(n_comittee_members):
        learners.append(ActiveLearner(estimator=estimator, X_training=X_train, y_training=y_train))
        
    # init committee
    committee = Committee(learner_list=learners, query_strategy=max_disagreement_sampling)

    unqueried_score = committee.score(X, y)
    print('Score over unqueried samples {:0.4f}'.format(unqueried_score))

    performance_history = []

    f1_score = 0
    index = 0
    while f1_score < 0.65:
        index += 1

        # get sample from pool
        query_idx, query_instance = committee.query(X_pool)

        # retrain comittee with new sample
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )

        # remove queried instance from pool
        X_pool = delete_rows_csr(X_pool, query_idx)
        y_pool = np.delete(y_pool, query_idx)

        y_pred = committee.predict(X)
        f1_score = metrics.f1_score(y, y_pred, average='micro')

        if index % 100 == 0:
            print('F1 score after {n} training samples: {f1:0.4f}'.format(n=index, f1=f1_score))

        # save accuracy score
        performance_history.append(f1_score)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(performance_history)
    return index


if __name__ == "__main__":
    run()