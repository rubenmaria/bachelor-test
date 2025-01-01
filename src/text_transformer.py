from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.typing import NDArray


class TextTransformer:
    __model = None

    def __new__(cls):
        if cls.__model is None:
            cls.__model = super(TextTransformer, cls).__new__(cls)
            cls.__model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls.__model
