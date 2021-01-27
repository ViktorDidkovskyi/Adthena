import pandas as pd

# using scikit learn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Preprocessor:
    """Class which helps in preprocessing train data"""

    def __init__(self, input_df: pd.DataFrame, split_train_test: bool = True):
        self._x_train = pd.DataFrame()
        self._x_test = pd.DataFrame()
        self._y_train = pd.DataFrame()
        self._y_test = pd.DataFrame()
        self.split_train_test = split_train_test
        self.data_processor(input_df)

    def data_processor(self, input_df: pd.DataFrame):
        """ Method which helps to preprocess the train file
        :param input_df: Training data
        :return:
        """

        shuffled_df = shuffle(input_df)
        # Features

        x = shuffled_df['norm_search_term']

        # Response
        y = shuffled_df['category']

        if self.split_train_test:
            self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(x, y, test_size=0.3, stratify=y)
        else:
            self._x_train = x.copy()
            self._y_train = y.copy()
            self._x_test = pd.DataFrame()
            self._y_test = pd.DataFrame()

    def get_data(self):
        return self._x_train, self._x_test, self._y_train, self._y_test

