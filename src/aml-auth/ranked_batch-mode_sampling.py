import pandas as pd
import scipy.sparse as sp
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
import numpy as np
from data_processing import get_fully_processed
import classifier_chains


""" For more information on Ranked batch-mode sampling you can read the following paper: 
Thiago N.C. Cardoso, Rodrigo M. Silva, Sérgio Canuto, Mirella M. Moro, Marcos A. Gonçalves. 
Ranked batch-mode active learning. Information Sciences, Volume 379, 2017, Pages 313-337.
https://www.sciencedirect.com/science/article/abs/pii/S0020025516313949
"""

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def run():

    books_df, genres_to_predict = get_fully_processed()
    X = books_df['book_description_processed']
    y_initial = books_df['major_genre']

    tfidf_initial = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_initial = tfidf_initial.fit_transform(X)


    # Isolate our examples for our labeled dataset.
    n_labeled_examples = X_initial.shape[0]
    training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=10)

    X_train = X_initial[training_indices, :]
    y_train = y_initial.iloc[training_indices]

    # Isolate the non-training examples we'll be querying.
    X_pool = delete_rows_csr(X_initial, training_indices)
    y_pool = y_initial.drop(training_indices)

    #tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    #X_train = tfidf.fit_transform(X_train)
    #X_pool = tfidf.transform(X_pool)

    logreg = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300)

    # Pre-set our batch sampling to retrieve 3 samples at a time.
    BATCH_SIZE = 3
    preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)

    # Specify our active learning model.
    learner = ActiveLearner(
        estimator=logreg,
        X_training=X_train,
        y_training=y_train,
        query_strategy=preset_batch
    )

    # Pool-based sampling
    N_RAW_SAMPLES = 20
    N_QUERIES = N_RAW_SAMPLES // BATCH_SIZE

    unqueried_score = learner.score(X_initial, y_initial)
    performance_history = [unqueried_score]

    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index, :], y_pool[query_index]
        #y = y.to_numpy()
        y = y.to_frame()
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool = delete_rows_csr(X_pool, training_indices)
        y_pool = y_pool.drop(query_index)

        # Calculate and report our model's accuracy.
        model_accuracy = learner.score(X_initial, y_initial)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)


if __name__ == "__main__":
    run()
