import gzip
import glob
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split

from data.sequence import DocumentsSequence
from config import (
    MAX_LEN, MAX_WORDS, NUM_CLASSES, CLASSES, DATASET_TEXT_COLUMN,
    DATASET_CLASS_COLUMN)


# API


def load_dataset(path):
    files_pattern = '{}/*.json.gz'.format(path)
    datasets = [gzip.open(f, 'r') for f in glob.glob(files_pattern)]
    for dataset in datasets:
        for line in dataset:
            yield eval(line)


def generate_train_data(dataset, log_dir, num_samples=120000):
    records = take_records(dataset, int(num_samples / NUM_CLASSES))
    data = process_records(records)
    vocab, documents = tokenize_documents(data, log_dir)
    labels = to_categorical(data['label'].values, num_classes=NUM_CLASSES)
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        documents, labels, test_size=0.25)
    print_dataset_info(vocab, documents, train_labels, val_labels)
    return vocab, (train_docs, train_labels), (val_docs, val_labels)


def generate_train_sequences(train_pair, val_pair):
    return DocumentsSequence(*train_pair), DocumentsSequence(*val_pair)


def generate_inference_sequence(text, tokenizer_path):
    tokenizer = Tokenizer()
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_LEN)


# Internal Functions


def process_records(records):
    data = pd.DataFrame(records)
    data = data[[DATASET_TEXT_COLUMN, DATASET_CLASS_COLUMN]]
    for label, clazz in enumerate(CLASSES):
        data.loc[data[DATASET_CLASS_COLUMN] == clazz, 'label'] = label

    return data.reindex(np.random.permutation(data.index))


def tokenize_documents(data, log_dir):
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(data[DATASET_TEXT_COLUMN].values)
    with open('{}/tokenizer.pickle'.format(log_dir), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequences = tokenizer.texts_to_sequences(data[DATASET_TEXT_COLUMN].values)
    documents = pad_sequences(sequences, maxlen=MAX_LEN)
    return tokenizer.word_index, documents


def take_records(dataset, quantity, classes=CLASSES, max_review_length=300):
    counters = [[c, 0] for c in classes]
    records = []
    while(not all([counter == quantity for _, counter in counters])):
        record = next(dataset)
        if len(record.get(DATASET_TEXT_COLUMN)) > max_review_length:
            continue

        should_add, class_index = should_add_record(record, counters, quantity)
        if should_add:
            records.append(record)
            counters[class_index][1] += 1

    return records


def should_add_record(record, counters, quantity):
    for i, [clazz, counter] in enumerate(counters):
        if record.get(DATASET_CLASS_COLUMN) == clazz and counter < quantity:
            return True, i

    return False, None


def print_dataset_info(vocab, documents, train_labels, val_labels):
    def print_class_distributions(labels):
        counts = np.count_nonzero((labels == [1. for _ in CLASSES]), axis=0)
        for label, count in enumerate(counts):
            print('Found {} {}s'.format(count, label))

    print('----------------------------')
    print('VOCAB SIZE: {}'.format(len(vocab) + 1))
    print('TOTAL SET LENGTH: {}'.format(len(documents)))
    print('TRAINING SET LENGTH: {}'.format(len(train_labels)))
    print('VAL SET LENGTH: {}'.format(len(val_labels)))
    print('TRAINING LABELS DISTRIBUTION')
    print_class_distributions(train_labels)
    print('VAL LABELS DISTRIBUTION')
    print_class_distributions(val_labels)
    print('----------------------------')
