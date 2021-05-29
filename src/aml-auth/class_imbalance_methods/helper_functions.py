from data_processing import get_processed_split, get_selected_genres, get_fully_processed


def get_baseline_split(representation=None):

    if representation == "bow":
        X_train, X_test, y_train, y_test = get_processed_split(genres_list=get_selected_genres(), vectorized="bow", multilabel=False)
    elif representation == "tf-idf":
        X_train, X_test, y_train, y_test = get_processed_split(genres_list=get_selected_genres(), vectorized="tf-idf", multilabel=False)
    else:
        raise ValueError("Invalid representation")

    return X_train, X_test, y_train, y_test


def get_fully_processed_books_df():
    books_df, genres_to_predict = get_fully_processed(genres_list=get_selected_genres(), multilabel=False)

    return books_df