import string
import re
import collections
from nltk import word_tokenize, download
from nltk.corpus import stopwords

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
    input_text = re.sub('[^a-zA-Z ?!]+', '', input_text)
    input_text = remove_stop_words(input_text)

    return input_text


def remove_punctuation(input_text):
    return input_text.translate(str.maketrans('', '', string.punctuation))


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
        books_df['primary_genres_list'] = books_df.apply(lambda book: list(set(book['primary_genres_list']).intersection(set(genres_to_keep))), axis=1)
        # filter out books with no genres left
        books_df = books_df[books_df['primary_genres_list'].map(lambda genre_list: len(genre_list)) > 0]

    elif genre_type == 'all':
        books_df['genres_list'] = books_df.apply(lambda book: list(set(book['genres_list']).intersection(set(genres_to_keep))), axis=1)
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


if __name__ == "__main__":
    # TEST
    from data_loader import read_goodreads_10k

    books_df = read_goodreads_10k()

    frequent_genres = get_n_most_frequent_genres(books_df, 'primary')

    print(frequent_genres)

    books_df_filtered = filter_out_genres(books_df, 'primary', frequent_genres)

    books_df_filtered = genres_to_onehot(books_df_filtered, 'primary', frequent_genres)

    print(books_df_filtered)

    # TEST text conditioning
    # for index, book in books_df.iterrows():
    #     print('========================')
    #     print(book['book_description'])
    #     print('------------------------')
    #     print(text_conditioning(book['book_description']))