import argparse

from model import SentimentModel
from data import load_dataset, generate_sequences
from config import DEFAULT_LOG_DIR, MAX_WORDS


NUM_TRAINING_SAMPLES = 120000


def train(dataset, logs):
    _, training_seq, validation_seq = generate_sequences(
        load_dataset(dataset), NUM_TRAINING_SAMPLES)
    model = SentimentModel(config={
        'vocab_size': MAX_WORDS,
        'log_dir': logs
    })
    model.train(training_seq, validation_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Attention-Based LSTM for sentiment analysis')
    parser.add_argument('command',
                        metavar='<command>',
                        help='`train` or `inference`')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOG_DIR,
                        metavar='/path/to/logs/',
                        help='Logs dir (default=/tmp/sentiment-analysis)')
    parser.add_argument('--dataset', required=False,
                        metavar='path to dataset',
                        help='Path to dataset directory')
    parser.add_argument('--text', required=False,
                        metavar='text to do inference with',
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    if args.command == 'train':
        assert args.dataset, 'Argument --dataset is required for training'
        train(args.dataset, args.logs)

    if args.command == 'inference':
        assert args.text, 'Argument --text is required for inference'
