from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class DomainFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Creates domain-informed features from building variables.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_data = X.copy()

        transformed_data["envelope_area"] = (
            transformed_data["wall_area"]
            + transformed_data["roof_area"]
        )

        transformed_data["glazing_height_interaction"] = (
            transformed_data["glazing_area"]
            * transformed_data["overall_height"]
        )

        transformed_data["compactness_height_interaction"] = (
            transformed_data["relative_compactness"]
            * transformed_data["overall_height"]
        )

        return transformed_data