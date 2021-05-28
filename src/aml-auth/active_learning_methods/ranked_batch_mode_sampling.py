import numpy as np
import time
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
from active_learning_methods.helper_functions import delete_rows_csr

""" For more information on Ranked batch-mode sampling you can read the following paper: 
Thiago N.C. Cardoso, Rodrigo M. Silva, Sérgio Canuto, Mirella M. Moro, Marcos A. Gonçalves. 
Ranked batch-mode active learning. Information Sciences, Volume 379, 2017, Pages 313-337.
https://www.sciencedirect.com/science/article/abs/pii/S0020025516313949
"""


def run(X_initial, y_initial, n_samples_for_initial, n_queries, batch_size, estimator):
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
    preset_batch = partial(uncertainty_batch_sampling, n_instances=batch_size)

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

    model_accuracy = initial_accuracy
    index = 0
    while model_accuracy < 0.55:
        index += 1
        query_index, query_instance = learner.query(X_pool)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index, :], y_pool[query_index]
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool = delete_rows_csr(X_pool, query_index)
        y_pool = np.delete(y_pool, query_index)

        # Calculate and report our model's accuracy.
        model_accuracy = learner.score(X_initial, y_initial)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)

    num_of_annotated_samples = index * batch_size
    print("--- %s seconds ---" % (time.time() - start_time))
    return num_of_annotated_samples


if __name__ == "__main__":
    run()
