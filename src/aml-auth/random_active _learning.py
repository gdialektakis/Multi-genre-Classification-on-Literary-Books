import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from functools import partial
from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
import numpy as np
from query_by_committee import prepare_data, create_random_pool_and_initial_sets
from ranked_batch_mode_sampling import delete_rows_csr


""" For more information on Ranked batch-mode sampling you can read the following paper: 
Thiago N.C. Cardoso, Rodrigo M. Silva, Sérgio Canuto, Mirella M. Moro, Marcos A. Gonçalves. 
Ranked batch-mode active learning. Information Sciences, Volume 379, 2017, Pages 313-337.
https://www.sciencedirect.com/science/article/abs/pii/S0020025516313949
"""

def run():

    X_initial, y_initial = prepare_data()
    X_train, y_train, X_pool, y_pool = create_random_pool_and_initial_sets(X_initial, y_initial, 100)

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

    initial_accuracy = learner.score(X_initial, y_initial)
    print("Initial Accuracy: ", initial_accuracy)
    performance_history = [initial_accuracy]

    # Random sampling
    N_QUERIES = 10000

    for index in range(N_QUERIES):
        query_index = np.random.choice(y_pool.shape[0], size=1, replace=False)

        # Teach our ActiveLearner model the random record it has been sampled.
        X, y = X_pool[query_index, :], y_pool[query_index]
        learner.teach(X=X, y=y)

        # fix this to retrain the classifier and not discard the previous training
        #clf.fit(X, y)

        # Remove the queried instance from the unlabeled pool.
        X_pool = delete_rows_csr(X_pool, query_index)
        y_pool = np.delete(y_pool, query_index)

        # Calculate and report our model's accuracy.
        model_accuracy = learner.score(X_initial, y_initial)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)


if __name__ == "__main__":
    run()
