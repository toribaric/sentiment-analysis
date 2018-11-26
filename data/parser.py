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
    MAX_LEN, MAX_WORDS, DATASET_TEXT_COLUMN, DATASET_SENTIMENT_COLUMN,
    DATASET_CATEGORY_COLUMN, NUM_SENTIMENTS, NUM_CATEGORIES,
    DATASET_SENTIMENT_VALUES, DATASET_CATEGORY_VALUES)


# API


def get_dataset_category(dataset_file_path):
    dataset_file_name = dataset_file_path.name.split('/')[-1]
    for category in DATASET_CATEGORY_VALUES:
        if category in dataset_file_name.lower():
            return category

    return None


def load_dataset(path):
    files_pattern = '{}/*.json.gz'.format(path)
    datasets = [gzip.open(f, 'r') for f in glob.glob(files_pattern)]
    for dataset in datasets:
        for i, line in enumerate(dataset):
            record = eval(line)
            record.update({'category': get_dataset_category(dataset)})
            yield record


def generate_train_data(dataset, log_dir, num_samples=120000):
    records = take_records(dataset, int(num_samples / NUM_SENTIMENTS))
    data = process_records(records)
    vocab, documents = tokenize_documents(data, log_dir)
    train_tuple, val_tuple = split_train_val_data(documents, data)
    return vocab, train_tuple, val_tuple


def generate_train_sequences(train_tuple, val_tuple):
    return DocumentsSequence(*train_tuple), DocumentsSequence(*val_tuple)


def generate_inference_sequence(text, tokenizer_path):
    tokenizer = Tokenizer()
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_LEN)


# Internal Functions


def take_records(dataset, quantity, sentiments=DATASET_SENTIMENT_VALUES,
                 max_review_length=300):
    counters = [[s, 0] for s in sentiments]
    records = []
    while(not all([counter == quantity for _, counter in counters])):
        record = next(dataset)
        if len(record.get(DATASET_TEXT_COLUMN)) > max_review_length:
            continue

        should_add, index = should_add_record(record, counters, quantity)
        if should_add:
            records.append(record)
            counters[index][1] += 1

    return records


def should_add_record(record, counters, quantity):
    for i, [sen, counter] in enumerate(counters):
        if record.get(DATASET_SENTIMENT_COLUMN) == sen and counter < quantity:
            return True, i

    return False, None


def process_records(records):
    data = pd.DataFrame(records)
    data = data[[DATASET_TEXT_COLUMN, DATASET_SENTIMENT_COLUMN,
                 DATASET_CATEGORY_COLUMN]]
    for label, sen in enumerate(DATASET_SENTIMENT_VALUES):
        data.loc[data[DATASET_SENTIMENT_COLUMN] == sen, 'sen_label'] = label

    for label, cat in enumerate(DATASET_CATEGORY_VALUES):
        data.loc[data[DATASET_CATEGORY_COLUMN] == cat, 'cat_label'] = label

    return data.reindex(np.random.permutation(data.index))


def tokenize_documents(data, log_dir):
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(data[DATASET_TEXT_COLUMN].values)
    with open('{}/tokenizer.pickle'.format(log_dir), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequences = tokenizer.texts_to_sequences(data[DATASET_TEXT_COLUMN].values)
    documents = pad_sequences(sequences, maxlen=MAX_LEN)
    return tokenizer.word_index, documents


def split_train_val_data(documents, data):
    sen_labels = to_categorical(
        data['sen_label'].values, num_classes=NUM_SENTIMENTS)
    cat_labels = to_categorical(
        data['cat_label'].values, num_classes=NUM_CATEGORIES)
    train_docs, val_docs, train_sen_labels, val_sen_labels, train_cat_labels, \
        val_cat_labels = train_test_split(
            documents, sen_labels, cat_labels, test_size=0.25)
    return (
        (train_docs, train_sen_labels, train_cat_labels),
        (val_docs, val_sen_labels, val_cat_labels)
    )


def print_dataset_info(vocab, documents, train_labels, val_labels):
    def print_class_distributions(labels):
        counts = np.count_nonzero(
            (labels == [1. for _ in DATASET_SENTIMENT_VALUES]), axis=0)
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
