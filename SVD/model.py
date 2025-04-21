from surprise.prediction_algorithms.matrix_factorization import SVD as SurpriseSVD
from surprise import Trainset
class SVD:
    def __init__(self, params):
        self._model = SurpriseSVD(**params)

    def train(self, matrix_interactions: Trainset):
        self._model.fit(matrix_interactions)
