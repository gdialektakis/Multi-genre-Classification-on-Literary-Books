import pandas as pd
import random
from data_processing import get_fully_processed, get_selected_genres
from nltk.corpus import wordnet
from random import sample


def get_unique_synonyms(token):
    list_of_synonyms = []

    for synonyms in wordnet.synsets(token):
        list_of_synonyms.extend([l.replace("_", " ").replace("-", " ").lower() for l in synonyms.lemma_names()])

        # if token in synonyms remove it
        if token in list_of_synonyms:
            list_of_synonyms.remove(token)
    
    return list(set(list_of_synonyms))
        

def get_synonym_tokens_replacement(token, tokens_to_replace_with=1):
    synonyms = get_unique_synonyms(token)

    if synonyms:
        random_synonyms = sample(synonyms, tokens_to_replace_with)

        return ' '.join(random_synonyms)
    else:
        return token


def synonym_replacement(description_text):
    augmented_description = []

    for token in description_text.split():
        augmented_description.append(get_synonym_tokens_replacement(token, tokens_to_replace_with=1))
     
    return ' '.join(augmented_description)


def unique_words_augmentation(description_text):
    return list(set(description_text.split()))


def random_mask(description_text):
    splitted_description_text = description_text.split()
    percentage_of_tokens_to_keep = random.uniform(0.8, 0.9)

    number_of_tokens_to_keep = int(len(splitted_description_text) * percentage_of_tokens_to_keep)

    return ' '.join(sample(splitted_description_text, number_of_tokens_to_keep)) 


def augment_dataset(books_df):
    class_distribution, genre_with_most_samples, n_samples_most = get_class_distribution(books_df)

    print(class_distribution)

    for major_genre, n_samples in class_distribution.items():
        samples_to_generate = min(int(n_samples*0.5), n_samples_most-n_samples)

        # get a random subset books for this genre
        books_of_this_genre = books_df[books_df['major_genre']==major_genre].sample(samples_to_generate)

        books_of_this_genre['book_description_processed'] = books_of_this_genre.apply(lambda x: synonym_replacement(x['book_description_processed']), axis=1)

        books_df = pd.concat([books_df, books_of_this_genre])
    
    return books_df


def get_class_distribution(books_df):
    class_distribution = books_df['major_genre'].value_counts().to_dict()

    genre_with_most_samples = max(class_distribution, key=class_distribution.get)
    n_samples = class_distribution[genre_with_most_samples]

    # remove genre with most samples from dictionary
    class_distribution.pop(genre_with_most_samples)

    return class_distribution, genre_with_most_samples, n_samples


if __name__ == "__main__":
    #print(get_unique_synonyms('small'))
    #print(get_synonym_tokens_replacement('small', tokens_to_replace_with=1))

    # books_df, genres_to_predict = get_fully_processed(genres_list=get_selected_genres())

    # print(books_df)
    # print(books_df.columns)

    
    # books_df = augment_dataset(books_df)

    # print(books_df)

    #distribution of multi label
    #print(books_df.groupby(genres_to_predict).size().reset_index().rename(columns={0:'count'})[genres_to_predict+['count']])

    text = 'transport maroon midway crosswise army for the liberation of rwanda recession'

    print(random_mask(text))

