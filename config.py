import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = '/tmp/sentiment-analysis'
MAX_LEN = 100
MAX_WORDS = 10000
BATCH_SIZE = 32
DATASET_TEXT_COLUMN = 'reviewText'
DATASET_SENTIMENT_COLUMN = 'overall'
DATASET_CATEGORY_COLUMN = 'category'
DATASET_SENTIMENT_VALUES = [1., 3., 5.]
PREDICTED_SENTIMENT_VALUES = ['negative', 'neutral', 'positive']
NUM_SENTIMENTS = len(DATASET_SENTIMENT_VALUES)
DATASET_CATEGORY_VALUES = [
    'books', 'beauty', 'cds_and_vinyl', 'electronics', 'grocery',
    'health', 'movies_and_tv', 'pet_supplies', 'sports_and_outdoors',
    'video_games'
]
PREDICTED_CATEGORY_VALUES = [
    'books', 'beauty', 'CD and vinyl', 'electronics', 'groceries', 'health',
    'movies and tv', 'pets', 'sports and outdoors', 'video games'
]
NUM_CATEGORIES = len(DATASET_CATEGORY_VALUES)
