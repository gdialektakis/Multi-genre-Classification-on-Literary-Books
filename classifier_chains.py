from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB

from data.data_loader import read_goodreads_10k
from data.data_processing import text_conditioning, get_n_most_frequent_genres, filter_out_genres, genres_to_onehot


def load_data():
    # LOAD DATA
    books_df = read_goodreads_10k()

    books_df['book_description_processed'] = books_df.apply(lambda book: text_conditioning(book['book_description']),
                                                            axis=1)
    classification_on = 'primary'

    genres_to_predict = get_n_most_frequent_genres(books_df, classification_on, n=10)

    books_df = filter_out_genres(books_df, classification_on, genres_to_predict)
    books_df = genres_to_onehot(books_df, classification_on, genres_to_predict)
    print(books_df)
    # SPLIT AND VECTORIZE
    train, test = train_test_split(books_df, test_size=0.25)
    return train, test, genres_to_predict


def vectorize(train ,test, genres_to_predict):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train['book_description_processed'])
    X_test = vectorizer.transform(test['book_description_processed'])
    y_train = train[genres_to_predict]
    y_test = test[genres_to_predict]
    return X_train, X_test, y_train, y_test



def classifier_chains(classifier):
    train, test, genres_to_predict = load_data()
    X_train, X_test, y_train, y_test = vectorize(train, test, genres_to_predict)
    chain = ClassifierChain(classifier, order='random', random_state=0)
    chain.fit(X_train, y_train)
    # chain.predict_proba(X_test)
    y_pred = chain.predict(X_test)
    return y_test, y_pred


def evaluate(y_test, y_pred):
    print('Accuracy score: {:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('Recall score: {:.3f}'.format(metrics.recall_score(y_test, y_pred, average='samples')))
    print('Precision score: {:.3f}'.format(metrics.precision_score(y_test, y_pred, average='samples')))
    print('F1 score: {:.3f}'.format(metrics.f1_score(y_test, y_pred, average='samples')))
    print('Hamming loss: {:.3f}'.format(metrics.hamming_loss(y_test, y_pred)))


if __name__ == "__main__":
    classifier = LogisticRegression(solver='lbfgs', random_state=0, max_iter=300)
    y_test, y_pred = classifier_chains(classifier)
    print('\n Classifier : {:}'.format(classifier))
    evaluate(y_test, y_pred)
    # Naive Bayes
    classifier = MultinomialNB()
    y_test, y_pred = classifier_chains(classifier)
    print('\n Classifier : {:}'.format(classifier))
    evaluate(y_test, y_pred)
