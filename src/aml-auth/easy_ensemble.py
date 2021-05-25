import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from data_processing import get_major_genre_split

def run():
    X_train, X_test, y_train, y_test = get_major_genre_split()
    a =1


if __name__ == "__main__":
    run()