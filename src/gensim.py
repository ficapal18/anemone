import gensim
from nltk.tokenize import word_tokenize

from src.similarity import Similarity

class GenSim(Similarity):

    def __init__(self):
        self.id = 0

    def tdfidf_similarity(self, gen_docs):
        return

