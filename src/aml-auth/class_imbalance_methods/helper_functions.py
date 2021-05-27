from data_processing import get_processed_split, label_encode_genres, get_selected_genres


def get_baseline_split(representation=None):
    genres_list = get_selected_genres()

    if representation == "bow":
        X_train, X_test, y_train, y_test = get_processed_split(genres_list=genres_list, vectorized="bow")
    elif representation == "tf-idf":
        X_train, X_test, y_train, y_test = get_processed_split(genres_list=genres_list, vectorized="tf-idf")
    else:
        raise ValueError("Invalid representation")

    y_train = label_encode_genres(y_train, genres_list)
    y_test = label_encode_genres(y_test, genres_list)

    y_train = y_train['major_genre'].to_frame()
    y_test = y_test['major_genre'].to_frame()

    return X_train, X_test, y_train, y_test