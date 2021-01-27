from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class TfidfLocVectorizer(TfidfVectorizer):

    def __init__(self, stop_words: str = 'english',
                 max_feat: int = None,
                 maxdf: float = 0.8,
                 mindf: float = 15,
                 n_gram_range: tuple = (1, 1)):
        """
        Constructor
        :param max_feat: maximum number of features for TF-IDF
        >>> tf = TfidfLocVectorizer(500)
        >>> tf.tfidf.max_features
        500
        """
        super().__init__()
        self.tfidf = TfidfVectorizer(stop_words=stop_words, strip_accents='ascii', max_df=maxdf,
                                     min_df=mindf, ngram_range=n_gram_range,
                                     max_features=max_feat)

    def fit_transform(self, raw_term: pd.DataFrame, y=None):
        """ Function called during training
        :param raw_term: dataframe with terms column
        :param y: expected output
        :return: vector representation

        """

        vec_rep = self.tfidf.fit_transform(raw_term)
        return vec_rep

    def transform(self, raw_term: pd.DataFrame, copy="deprecated"):
        """ Function called during prediction
        :param raw_term: dataframe with terms column
        :param copy: copy results
        :return: vector representation

        """
        vec_rep = self.tfidf.transform(raw_term)
        return vec_rep

