import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = '/tmp/sentiment-analysis'
MAX_LEN = 100
MAX_WORDS = 10000
BATCH_SIZE = 32
NUM_CLASSES = 3
CLASSES = [1., 3., 5.]
SENTIMENTS = ['negative', 'neutral', 'positive']
