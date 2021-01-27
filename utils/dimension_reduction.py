from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD


class TruncatedLocSVD(TransformerMixin):
    """ Class to reduce the dimension using SVD

    """

    def __init__(self, optimum_n_components: int = None,
                 total_variance: float = 0.8):
        """

        :param optimum_n_components: the number of components ( output features number). If it is None, than find
        the optimum number of components.
        :param total_variance: the target goal level of explained variance
        """

        self.optimum_n_components = optimum_n_components
        self.total_variance = total_variance

    def transform(self, X, **transform_params):


        return self.tsvd.transform(X)

    def fit(self, X, y=None, **fit_params):

        return self

    def fit_transform(self, X, y=None, **fit_params):

        self.__optimum_components(X)

        return self.tsvd.fit_transform(X)

    def get_params(self, deep=True):
        return {}

    def __optimum_components(self, X):

        if (not self.optimum_n_components) or X.shape[1] <= self.optimum_n_components:
            tsvd = TruncatedSVD(n_components=X.shape[1] - 1)
            tsvd.fit(X)
            tsvd_var_ratios = tsvd.explained_variance_ratio_
            self.optimum_n_components = self.select_n_components(tsvd_var_ratios, self.total_variance)
        self.tsvd = TruncatedSVD(n_components=self.optimum_n_components)

    @staticmethod
    def select_n_components(var_ratio: list, goal_var: float) -> int:
        """
        :param var_ratio: ratio dependent on the dimension
        :param goal_var: target goal level of explained variance
        :return:
        """
        # Set initial variance explained so far
        total_variance = 0.0

        # Set initial number of features
        n_components = 0

        # For the explained variance of each feature:
        for explained_variance in var_ratio:

            # Add the explained variance to the total
            total_variance += explained_variance

            # Add one to the number of components
            n_components += 1

            # If we reach our goal level of explained variance
            if total_variance >= goal_var:
                # End the loop
                break

        # Return the number of components
        return n_components
