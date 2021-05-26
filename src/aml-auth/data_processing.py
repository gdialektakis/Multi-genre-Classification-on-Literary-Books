import re
import collections
from nltk import download
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

from data.data_loader import read_goodreads_10k

download('stopwords')


def text_conditioning(input_text):
    input_text = input_text.lower()
    input_text = re.sub(r"what's", 'what is ', input_text)
    input_text = re.sub(r"\'ve", ' have ', input_text)
    input_text = re.sub(r"can't", 'cannot ', input_text)
    input_text = re.sub(r"n't", ' not ', input_text)
    input_text = re.sub(r"i'm", 'i am ', input_text)
    input_text = re.sub(r"\'re", ' are ', input_text)
    input_text = re.sub(r"\'d", ' would ', input_text)
    input_text = re.sub(r"\'ll", ' will ', input_text)
    input_text = remove_punctuation(input_text)
    input_text = re.sub('[^a-zA-Z]+', ' ', input_text)
    input_text = remove_stop_words(input_text)

    return input_text


def remove_punctuation(input_text):
    return re.sub(r'[^\w\s]', ' ', input_text)


def whitespaces_conditioning(input_text):
    return ' '.join(input_text.split())


def remove_stop_words(tokenized_input_text):
    stop_words = stopwords.words('english')

    return ' '.join(token for token in tokenized_input_text.split() if not token in stop_words)


def get_n_most_frequent_genres(books_df, genre_type, n=10):
    if genre_type == 'primary':
        primary_genre_counter = dict(collections.Counter(books_df['primary_genres_list'].sum()))
        return list(dict(sorted(primary_genre_counter.items(), key=lambda item: item[1], reverse=True)[:n]).keys())

    elif genre_type == 'all':
        all_genre_counter = dict(collections.Counter(books_df['genres_list'].sum()))
        return list(dict(sorted(all_genre_counter.items(), key=lambda item: item[1], reverse=True)[:n]).keys())
    else:
        raise ValueError('Wrong genre type input')


def filter_out_genres(books_df, genre_type, genres_to_keep):
    if genre_type == 'primary':
        books_df['primary_genres_list'] = books_df.apply(
            lambda book: list(set(book['primary_genres_list']).intersection(set(genres_to_keep))), axis=1)
        # filter out books with no genres left
        books_df = books_df[books_df['primary_genres_list'].map(lambda genre_list: len(genre_list)) > 0]

    elif genre_type == 'all':
        books_df['genres_list'] = books_df.apply(
            lambda book: list(set(book['genres_list']).intersection(set(genres_to_keep))), axis=1)
        # filter out books with no genres left
        books_df = books_df[books_df['primary_genres_list'].map(lambda genre_list: len(genre_list)) > 0]

    else:
        raise ValueError('Wrong genre type input')

    return books_df


def genres_to_onehot(books_df, genre_type, genres_to_predict):
    if genre_type == 'primary':
        for genre in genres_to_predict:
            books_df[genre] = 0

        for genre in genres_to_predict:
            books_df[genre] = books_df['primary_genres_list'].apply(lambda x: 1 if genre in x else 0)

    return books_df


def major_genre_label_encoding(books_df):
    le = LabelEncoder()

    books_df['major_genre'] = le.fit_transform(books_df['major_genre'])

    return books_df


def get_genre_label(row, genre_list):
    for counter, genre in enumerate(genre_list):
        if row[genre] == 1:
            return counter

    print(row)
    print(row.values)


def label_encode_genres(books_df, genre_list):
    books_df['genre_label'] = books_df.apply(lambda row: get_genre_label(row, genre_list), axis=1)

    return books_df


def get_fully_processed(classification_on="primary", num_of_genres=10, genres_list=None):
    books_df = read_goodreads_10k()

    books_df['book_description_processed'] = books_df.apply(lambda book: text_conditioning(book['book_description']),
                                                            axis=1)

    if not genres_list:
        genres_to_predict = get_n_most_frequent_genres(books_df, classification_on, n=num_of_genres)
    else:
        genres_to_predict = genres_list

    books_df = filter_out_genres(books_df, classification_on, genres_to_predict)
    books_df = genres_to_onehot(books_df, classification_on, genres_to_predict)
    books_df = major_genre_label_encoding(books_df)

    return books_df, genres_to_predict


def get_processed_split(classification_on="primary", num_of_genres=10, genres_list=None,
                        test_size=0.25, vectorized='tf-idf', max_features=1000, ngram_range=(1, 2)):
    books_df, genres_to_predict = get_fully_processed(classification_on=classification_on,
                                                      num_of_genres=num_of_genres,
                                                      genres_list=genres_list)
    train, test = train_test_split(books_df, test_size=test_size)

    if vectorized:
        if vectorized == "tf-idf":
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        elif vectorized == "bow":
            vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range)
        else:
            raise ValueError("Invalid vectorized argument, please choose on of [tf-idf, bow, None]")

        X_train = vectorizer.fit_transform(train['book_description_processed'])
        X_test = vectorizer.transform(test['book_description_processed'])
    else:
        X_train = train['book_description_processed']
        X_test = test['book_description_processed']

    y_train = train[genres_to_predict]
    y_test = test[genres_to_predict]

    return X_train, X_test, y_train, y_test


def get_major_genre_split(test_size=0.25, max_features=1000, ngram_range=(1, 2)):
    books_df, genres_to_predict = get_fully_processed()

    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(books_df['book_description_processed'])
    y = books_df['major_genre'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def run():
    # TEST
    books_df = read_goodreads_10k()

    frequent_genres = get_n_most_frequent_genres(books_df, 'primary', 24)

    print(f"frequent genres: {frequent_genres}")

    frequent_genres = ['Romance', 'Adventure', 'Audiobook', 'Young Adult', 'Space', 'Historical', 'Adult',
                       'Speculative Fiction', 'War', 'Apocalyptic']

    books_df_filtered = filter_out_genres(books_df, 'primary', frequent_genres)

    books_df_filtered = genres_to_onehot(books_df_filtered, 'primary', frequent_genres)

    print(books_df_filtered)

    # TEST text conditioning
    for index, book in books_df.iterrows():
        print('========================')
        print(book['book_description'])
        print('------------------------')
        print(text_conditioning(book['book_description']))


if __name__ == "__main__":
    run()
