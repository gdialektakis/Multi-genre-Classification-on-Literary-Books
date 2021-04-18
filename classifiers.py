import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

from data.data_loader import read_goodreads_10k
from data.data_processing import text_conditioning, get_n_most_frequent_genres, filter_out_genres, genres_to_onehot


# LOAD DATA
books_df = read_goodreads_10k()

books_df['book_description_processed'] = books_df.apply(lambda book: text_conditioning(book['book_description']), axis=1)

classification_on = 'primary'

genres_to_predict = get_n_most_frequent_genres(books_df, classification_on, n=10)

books_df = filter_out_genres(books_df, classification_on, genres_to_predict)
books_df = genres_to_onehot(books_df, classification_on, genres_to_predict)

print(books_df)

# SPLIT AND VECTORIZE
train, test = train_test_split(books_df, test_size=0.25)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

X_train = vectorizer.fit_transform(train['book_description_processed'])
X_test = vectorizer.transform(test['book_description_processed'])

## ONE VS REST
for genre in genres_to_predict:
    print('OneVsRest classification for genre: {}'.format(genre))

    model = OneVsRestClassifier(LogisticRegression())
    model.fit(X_train, train[genre])

    y_pred = model.predict(X_test)


    print('Accuracy score: {:.3f}'.format(metrics.accuracy_score(test[genre], y_pred)))
    print('Recall score: {:.3f}'.format(metrics.recall_score(test[genre], y_pred)))
    print('Precision score: {:.3f}'.format(metrics.precision_score(test[genre], y_pred)))
    print('F1 score: {:.3f}'.format(metrics.f1_score(test[genre], y_pred)))

# MULTIOUTPUT CLASSIFIER
model = MultiOutputClassifier(LogisticRegression())
model.fit(X_train, train[genres_to_predict])

y_pred = model.predict(X_test)

print(model.score(X_test, test[genres_to_predict]))

auc_y1 = roc_auc_score(test[genres_to_predict].values[:,0],y_pred[:,0])
auc_y2 = roc_auc_score(test[genres_to_predict].values[:,1],y_pred[:,1])

 
print("ROC AUC y1: %.4f, y2: %.4f" % (auc_y1, auc_y2))

#### MORE
# https://scikit-learn.org/stable/modules/multiclass.html
# https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html
# EXAMPLE https://www.datatechnotes.com/2020/03/multi-output-classification-with-multioutputclassifier.html