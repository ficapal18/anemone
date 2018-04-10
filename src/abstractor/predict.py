from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np
import pandas as pd

import time
import pickle

# import nltk
# nltk.download('punkt')

from ml_component import Ml_component
import model

class Main(Ml_component):

    def __init__(self):
        model = model.Model()


Main()