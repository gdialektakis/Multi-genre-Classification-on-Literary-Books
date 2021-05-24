import scipy.sparse as sp
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from data_processing import get_fully_processed


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


def train_classifier(X_train, y_train):
    """
        Train a classifier on the initial labeled data.
        """
    logreg = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300, multi_class='multinomial')
    logreg.fit(X_train, y_train)
    return logreg


def evaluate_clasifier(X, y, clf):
    y_pred = clf.predict(X)
    accuracy_score = metrics.accuracy_score(y, y_pred)
    #print(f"Accuracy score: {accuracy_score:.2f}")
    return accuracy_score


def run():

    books_df, genres_to_predict = get_fully_processed()
    X = books_df['book_description_processed']
    y_initial = books_df['major_genre'].values

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_initial = tfidf.fit_transform(X)

    # Isolate our examples for our labeled dataset.
    n_labeled_examples = X_initial.shape[0]
    training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=10)

    X_train = X_initial[training_indices, :]
    y_train = y_initial[training_indices]

    # Isolate the non-training examples we'll be querying.
    X_pool = delete_rows_csr(X_initial, training_indices)
    y_pool = np.delete(y_initial, training_indices)

    # train the classifier on the initial labeled data
    clf = train_classifier(X_train, y_train)


    initial_accuracy = evaluate_clasifier(X_initial, y_initial, clf)
    print("Initial Accuracy: ", initial_accuracy)
    performance_history = [initial_accuracy]

    # Random sampling
    N_QUERIES = 10000

    for index in range(N_QUERIES):
        query_index = np.random.choice(y_pool.shape[0], size=1, replace=False)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index, :], y_pool[query_index]
        # fix this to retrain the classifier and not discard the previous training
        clf.fit(X, y)

        # Remove the queried instance from the unlabeled pool.
        X_pool = delete_rows_csr(X_pool, query_index)
        y_pool = np.delete(y_pool, query_index)

        # Calculate and report our model's accuracy.
        model_accuracy = evaluate_clasifier(X_initial, y_initial, clf)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)


if __name__ == "__main__":
    run()
