import pandas as pd
import json
import re
import collections
import matplotlib.pyplot as plt
from os import path, listdir


def genre_string_to_list(genre_string):
    # replace single quote with double
    genre_string = genre_string.replace("'", '"')
    genre_json = json.loads(genre_string)

    genre_labels = list(genre_json.keys())
    
    # remove trailing spaces
    genre_labels = list(map(str.strip, genre_labels))

    return genre_labels


def generate_primary_genres(label_list):
    return list(set([re.sub(r'\s\(.*', '', label) for label in label_list]))


def get_description_length(description_text):
    return len(description_text.split())


def read_goodreads_10k():
    data_directory = get_data_path()

    list_of_csv_files = files = list(filter(lambda f: f.endswith('.csv'), listdir(data_directory)))

    # init books df
    books_df = pd.DataFrame(None)
    for file_name in list_of_csv_files:
        df = pd.read_csv(path.join(data_directory, file_name))
        df['source'] = file_name[3:-4]

        books_df = pd.concat([books_df, df])

    books_df.columns = map(str.lower, books_df.columns)

    # drop duplicates
    books_df.drop_duplicates(subset='url', keep='first', inplace=False)

    # keep english only books
    books_df = books_df[books_df['edition_language'] == 'English']

    # filter out books with null description
    books_df = books_df[books_df['book_description'].notnull()]

    # transform genres json column to list
    books_df['genres_list'] = books_df.apply(lambda row: genre_string_to_list(row['genres']), axis=1)
    books_df['primary_genres_list'] = books_df.apply(lambda row: generate_primary_genres(row['genres_list']), axis=1)
    books_df['description_length'] = books_df.apply(lambda book: get_description_length(book['book_description']), axis=1)
    
    books_df = books_df[books_df['description_length'] > 10]

    books_df.drop(columns=['genres', 'url', 'edition_language', 'original_book_title'], inplace=True)

    return books_df.reset_index(drop=True)


def get_data_path():
    return path.join(path.abspath(path.dirname(__file__)), 'goodreads_10k')


if __name__ == "__main__":
    # TEST
    books_df = read_goodreads_10k()
    print(books_df)
    print(books_df.columns)