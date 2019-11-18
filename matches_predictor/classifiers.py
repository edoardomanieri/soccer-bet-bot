from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def xgb(train_X, train_y, test_X, test_y):
    xgb = XGBClassifier(n_estimators=2000)
    xgb.fit(train_X, train_y)
    predictions = xgb.predict(test_X)
    return xgb, accuracy_score(test_y, predictions)
