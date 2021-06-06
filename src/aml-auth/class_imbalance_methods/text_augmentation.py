import pandas as pd
import random
from nltk.corpus import wordnet
from random import sample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_unique_synonyms(token):
    list_of_synonyms = []

    # if token doesn exist in wordnet return it as it is
    if not wordnet.synsets(token):
        return token

    # collect all synonyms found in wordnet
    for synonyms in wordnet.synsets(token):
        list_of_synonyms.extend([l.replace('_', ' ').replace('-', ' ').lower() for l in synonyms.lemma_names()])

    # if token in synonyms remove it
    if token in list_of_synonyms:
        list_of_synonyms.remove(token)
    
    return list(set(list_of_synonyms))
        

def get_synonym_tokens_replacement(token, n_tokens_to_replace_with=1):
    # get all synonyms of a token
    synonyms = get_unique_synonyms(token)

    if synonyms:
        # make a random selection of n_tokens_to_replace_with from the synonyms found
        random_synonyms = sample(synonyms, n_tokens_to_replace_with)

        return ' '.join(random_synonyms)
    else:
        return token


def synonym_replacement(description_text, n_tokens_to_replace_with=1):
    augmented_description = []

    # for each token in the original text find a number of tokens (n_tokens_to_replace_with) synonyms if exist
    for token in description_text.split():
        augmented_description.append(get_synonym_tokens_replacement(token, n_tokens_to_replace_with))
     
    return ' '.join(augmented_description)


def unique_words(description_text):
    # get all unique tokens found in description text 
    return ' '.join(list(set(description_text.split())))


def random_mask(description_text):
    splitted_description_text = description_text.split()

    # randomly keep 80-100 percent of original tokens
    percentage_of_tokens_to_keep = random.uniform(0.8, 1)
    number_of_tokens_to_keep = int(len(splitted_description_text) * percentage_of_tokens_to_keep)

    return ' '.join(sample(splitted_description_text, number_of_tokens_to_keep)) 


def augment_dataset(books_df):
    class_distribution, genre_with_most_samples, n_samples_most = get_class_distribution(books_df)

    # iterate through under represented class
    for major_genre, n_samples in class_distribution.items():
        # get number of samples to augment
        samples_to_generate = min(int(n_samples), n_samples_most-n_samples)

        # get a random subset books for this genre
        books_of_this_genre = books_df[books_df['major_genre']==major_genre].sample(samples_to_generate)

        books_of_this_genre_synonyms = books_of_this_genre.copy()
        books_of_this_genre_synonyms['book_description_processed'] = books_of_this_genre.apply(lambda x: synonym_replacement(x['book_description_processed']), axis=1)

        books_of_this_genre_unique = books_of_this_genre.copy()
        books_of_this_genre_unique['book_description_processed'] = books_of_this_genre.apply(lambda x: unique_words(x['book_description_processed']), axis=1)
        
        books_of_this_genre_mask = books_of_this_genre.copy()
        books_of_this_genre_mask['book_description_processed'] = books_of_this_genre.apply(lambda x: random_mask(x['book_description_processed']), axis=1)

        books_df = pd.concat([books_df, books_of_this_genre_synonyms, books_of_this_genre_unique, books_of_this_genre_mask])
        
    return books_df


def get_class_distribution(books_df):
    # get number of samples per class
    class_distribution = books_df['major_genre'].value_counts().to_dict()

    # find most represented class
    genre_with_most_samples = max(class_distribution, key=class_distribution.get)
    n_samples = class_distribution[genre_with_most_samples]

    # remove genre with most samples from dictionary
    class_distribution.pop(genre_with_most_samples)

    return class_distribution, genre_with_most_samples, n_samples


def run(books_df, representation):
    import nltk
    nltk.download('wordnet')

    print("######################")
    print("Text Augmentation")
    print("######################")
    print("\n")

    # augment dataset
    books_df = augment_dataset(books_df)

    train, test = train_test_split(books_df, test_size=0.25)

    # vectorize book description text
    if representation == "tf-idf":
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    elif representation == "bow":
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    else:
        raise ValueError("Invalid vectorized argument, please choose on of [tf-idf, bow, None]")

    X_train = vectorizer.fit_transform(train['book_description_processed'])
    X_test = vectorizer.transform(test['book_description_processed'])

    y_train = train['major_genre'].values
    y_test = test['major_genre'].values

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # test
    print(get_unique_synonyms('small'))



