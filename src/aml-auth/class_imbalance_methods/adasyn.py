from imblearn.over_sampling import ADASYN


def run(X_train, y_train):

    adasyn_sampler = ADASYN(sampling_strategy='not majority')
    X_adasyn, y_adasyn = adasyn_sampler.fit_resample(X_train, y_train)
    print("######################")
    print("ADASYN")
    print("######################")
    print("\n")
    return X_adasyn, y_adasyn


if __name__ == "__main__":
    run()
