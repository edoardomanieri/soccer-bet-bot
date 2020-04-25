import abc


class Model(metaclass=abc.ABCMeta):
    """
    Abstract class that is the super class of all models
    """

    @abc.abstractmethod
    def preprocess_train(self):
        pass

    @abc.abstractmethod
    def preprocess_input(self):
        pass

    @abc.abstractmethod
    def build_model(self, params):
        pass

    @abc.abstractmethod
    def train(self, model, train_X, train_y):
        pass

    @abc.abstractmethod
    def save_model(self):
        pass

    @abc.abstractmethod
    def get_predict_proba(self):
        pass
