from numpy.typing import ArrayLike
import implicit
from scipy.sparse import csr_matrix
from typing import Union

class AlternatingLeastSquares:
    def __init__(self, params: dict):
        self._model = implicit.als.AlternatingLeastSquares(**params)

    def train(self, matrix_interactions, show_progress=True):
        self._model.fit(matrix_interactions, show_progress=show_progress)

    def predict(self, userid: Union[int, ArrayLike] , user_items: csr_matrix, top_k):
        recommended_items = self._model.recommend(user_item=userid, user_items=user_items, N=top_k)

        return recommended_items
