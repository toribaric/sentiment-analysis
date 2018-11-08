import os
import datetime
import argparse

from model import SentimentModel
from data import (
    load_dataset, generate_train_sequences, generate_inference_sequence)
from config import DEFAULT_LOG_DIR, MAX_WORDS


NUM_TRAINING_SAMPLES = 450000


def create_model(log_dir=None):
    return SentimentModel(config={
        'vocab_size': MAX_WORDS,
        'log_dir': log_dir
    })


def train(dataset, logs):
    log_dir = '{}/iter_{:%Y%m%dT%H%M%S}'.format(
        args.logs, datetime.datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    _, training_seq, validation_seq = generate_train_sequences(
        load_dataset(dataset), log_dir, NUM_TRAINING_SAMPLES)
    model = create_model(log_dir)
    model.train(training_seq, validation_seq)


def inference(weights_path, tokenizer_path, text):
    model = create_model()
    model.load_weights(weights_path)
    sentiment = model.analyze(
        generate_inference_sequence(text, tokenizer_path))
    print('Text sentiment is {}'.format(sentiment))


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
    parser.add_argument('--weights', required=False,
                        metavar='path to saved weights',
                        help='Path to saved .h5 weights')
    parser.add_argument('--tokenizer', required=False,
                        metavar='path to saved tokenizer',
                        help='Path to saved tokenizer')
    parser.add_argument('--text', required=False,
                        metavar='text to analyse',
                        help='Text to analyse')
    args = parser.parse_args()

    if args.command == 'train':
        assert args.dataset, 'Argument --dataset is required for training'
        train(args.dataset, args.logs)

    if args.command == 'inference':
        assert args.weights, 'Argument --weights is required for inference'
        assert args.tokenizer, 'Argument --tokenizer is required for inference'
        assert args.text, 'Argument --text is required for inference'
        inference(args.weights, args.tokenizer, args.text)
