from surprise import SVDpp as SurpriseSVD


class SVDpp:
    def __init__(self, params):
        self._model = SurpriseSVD(**params)

    def train(self, matrix_interactions):
        self._model.fit(matrix_interactions)
