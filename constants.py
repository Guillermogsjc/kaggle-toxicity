import pandas as pd
import os


DATA_FOLDER = 'data'
DATA_TOOLS_FOLDER = 'data_tools'
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'results'

STORE_PATH = os.path.join(DATA_FOLDER, 'store.h5')

OBJECTIVE_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300