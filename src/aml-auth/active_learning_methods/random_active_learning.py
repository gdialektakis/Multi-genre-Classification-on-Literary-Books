import time
from functools import partial
from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
import numpy as np
from active_learning_methods.helper_functions import delete_rows_csr
from sklearn import metrics

""" For more information on Ranked batch-mode sampling you can read the following paper: 
Thiago N.C. Cardoso, Rodrigo M. Silva, Sérgio Canuto, Mirella M. Moro, Marcos A. Gonçalves. 
Ranked batch-mode active learning. Information Sciences, Volume 379, 2017, Pages 313-337.
https://www.sciencedirect.com/science/article/abs/pii/S0020025516313949
"""


def run(X_initial, y_initial, n_samples_for_initial, n_queries, estimator):
    np.random.seed(0)
    start_time = time.time()
    # Isolate our examples for our labeled dataset.
    n_labeled_examples = X_initial.shape[0]
    training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=n_samples_for_initial)

    X_train = X_initial[training_indices, :]
    y_train = y_initial[training_indices]

    # Isolate the non-training examples we'll be querying.
    X_pool = delete_rows_csr(X_initial, training_indices)
    y_pool = np.delete(y_initial, training_indices)
    # Pre-set our batch sampling to retrieve 3 samples at a time.
    BATCH_SIZE = 3
    preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)

    # Specify our active learning model.
    learner = ActiveLearner(
        estimator=estimator,
        X_training=X_train,
        y_training=y_train,
        query_strategy=preset_batch
    )

    initial_accuracy = learner.score(X_initial, y_initial)
    print("Initial Accuracy: ", initial_accuracy)
    performance_history = [initial_accuracy]

    f1_score = 0
    index = 0
    while f1_score < 0.65:
        index += 1
        query_index = np.random.choice(y_pool.shape[0], size=1, replace=False)

        # Teach our ActiveLearner model the random record it has been sampled.
        X, y = X_pool[query_index, :], y_pool[query_index]
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool = delete_rows_csr(X_pool, query_index)
        y_pool = np.delete(y_pool, query_index)

        # Calculate and report our model's f1_score.
        y_pred = learner.predict(X_initial)
        f1_score = metrics.f1_score(y_initial, y_pred, average='micro')

        if index % 100 == 0:
            print('F1 score after {n} training samples: {f1:0.4f}'.format(n=index, f1=f1_score))

        # Save our model's performance for plotting.
        performance_history.append(f1_score)

    print("--- %s seconds ---" % (time.time() - start_time))
    return index


if __name__ == "__main__":
    run()
