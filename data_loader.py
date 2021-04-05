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


def read_goodreads_10k():
    data_directory = path.join('data', 'goodreads_10k')

    list_of_csv_files = files = list(filter(lambda f: f.endswith('.csv'), listdir(data_directory)))

    # init books df
    books_df = pd.DataFrame(None)
    for file_name in list_of_csv_files:
        df = pd.read_csv(path.join(data_directory, file_name))
        df['source'] = file_name

        books_df = pd.concat([books_df, df])

    # keep english only books
    books_df = books_df[books_df['Edition_Language'] == 'English']

    # transform genres json column to list
    books_df['genres_list'] = books_df.apply(lambda row: genre_string_to_list(row['Genres']), axis=1)

    return books_df


def plot_label_distribution(df):
    all_instances_labels = df['genres_list'].sum()
    print(len(all_instances_labels))

    all_instances_unique_labels = list(set(all_instances_labels))
    print(len(all_instances_unique_labels))

    primary_labels = [re.sub(r'\s\(.*', '', label) for label in all_instances_labels]

    label_counter = dict(collections.Counter(all_instances_labels))
    # sort counter's dict
    label_counter = dict(sorted(label_counter.items(), key=lambda item: item[1], reverse=True)[:100])

    primary_label_counter = dict(collections.Counter(primary_labels))

    print(len(primary_label_counter.keys()))
    # sort counter's dict
    primary_label_counter = dict(sorted(primary_label_counter.items(), key=lambda item: item[1], reverse=True)[:100])


    fig = plt.figure(constrained_layout=True, figsize=(16,12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.title.set_text('All label distribution')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Occurence')
    plt.xticks(rotation=90)
    plt.bar(list(label_counter.keys()), list(label_counter.values()))

    ax2 = fig.add_subplot(gs[1, :])
    ax2.title.set_text('Primary label distribution')
    ax2.set_xlabel('Label')
    ax2.set_ylabel('Occurence')
    plt.xticks(rotation=90)
    plt.bar(list(primary_label_counter.keys()), list(primary_label_counter.values()))

    plt.show()


if __name__ == "__main__":
    books_df = read_goodreads_10k() # book titles also require parsing
    print(books_df)

    #plot_label_distribution(books_df)