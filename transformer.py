


class STransformer:

    def __init__(self):
        self.__model = SentenceTransformer('all-MiniLM-L6-v2')


    def text_to_embedding(self, function: str) -> NDArray:
        embedding = np.array(
            self.__model.encode(function, convert_to_numpy=True)
        )
        return embedding
