def train_model(clf, params, test_X=None, test_y=None):
    """
    Create model and save it with joblib
    """
    train_X, train_y = clf.preprocess_train()
    model = clf.build_model(**params)
    clf.train(model, train_X, train_y, test_X, test_y)
    clf.save_model(model)
