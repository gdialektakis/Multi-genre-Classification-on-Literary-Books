from imblearn.over_sampling import SMOTE


def run(X_train, y_train):
    smote_sampler = SMOTE(sampling_strategy='not majority', n_jobs=-1)
    X_smote, y_smote = smote_sampler.fit_resample(X_train, y_train)
    print("######################")
    print("SMOTE")
    print("######################")
    print("\n")
    return X_smote, y_smote


if __name__ == "__main__":
    run()
