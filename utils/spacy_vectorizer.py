from sklearn.base import BaseEstimator, TransformerMixin


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):

        return [self.nlp(text).vector for text in X]
