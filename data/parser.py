import gzip
import glob
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split

from data.sequence import DocumentsSequence
from config import MAX_LEN, MAX_WORDS, NUM_CLASSES


# API


def load_dataset(path):
    files_pattern = '{}/*.json.gz'.format(path)
    datasets = [gzip.open(f, 'r') for f in glob.glob(files_pattern)]
    for dataset in datasets:
        for line in dataset:
            yield eval(line)


def generate_train_sequences(dataset, log_dir, num_samples=120000):
    records = take_records(dataset, int(num_samples / NUM_CLASSES))
    data = process_records(records)
    vocab_size, documents = tokenize_documents(data, log_dir)
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        documents, data['label'].values, test_size=0.25)

    print_dataset_info(vocab_size, documents, train_labels, val_labels)

    return (vocab_size, DocumentsSequence(train_docs, train_labels),
            DocumentsSequence(val_docs, val_labels))


def generate_inference_sequence(text, tokenizer_path):
    tokenizer = Tokenizer()
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_LEN)


# Internal Functions


def process_records(records):
    data = pd.DataFrame(records)
    data = data[['reviewText', 'overall']]
    data.loc[data['overall'] == 1.0, 'label'] = 0
    data.loc[data['overall'] == 3.0, 'label'] = 1
    data.loc[data['overall'] == 5.0, 'label'] = 2
    return data.reindex(np.random.permutation(data.index))


def tokenize_documents(data, log_dir):
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(data['reviewText'].values)
    with open('{}/tokenizer.pickle'.format(log_dir), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequences = tokenizer.texts_to_sequences(data['reviewText'].values)
    vocab_size = len(tokenizer.word_index) + 1
    documents = pad_sequences(sequences, maxlen=MAX_LEN)
    return vocab_size, documents


def take_records(dataset, quantity, classes_to_take=[1., 3., 5.],
                 max_review_length=300):
    counters = [[c, 0] for c in classes_to_take]
    records = []
    while(not all([counter == quantity for _, counter in counters])):
        record = next(dataset)
        if len(record.get('reviewText')) > max_review_length:
            continue

        should_add, class_index = should_add_record(record, counters, quantity)
        if should_add:
            records.append(record)
            counters[class_index][1] += 1

    return records


def should_add_record(record, counters, quantity):
    for i, [clazz, counter] in enumerate(counters):
        if record.get('overall') == clazz and counter < quantity:
            return True, i

    return False, None


def print_dataset_info(vocab_size, documents, train_labels, val_labels):
    def print_class_distributions(labels):
        print('Found {} 0s'.format(np.count_nonzero(labels == 0)))
        print('Found {} 1s'.format(np.count_nonzero(labels == 1)))
        print('Found {} 2s'.format(np.count_nonzero(labels == 2)))

    print('----------------------------')
    print('VOCAB SIZE: {}'.format(vocab_size))
    print('TOTAL SET LENGTH: {}'.format(len(documents)))
    print('TRAINING SET LENGTH: {}'.format(len(train_labels)))
    print('VAL SET LENGTH: {}'.format(len(val_labels)))
    print('TRAINING LABELS DISTRIBUTION')
    print_class_distributions(train_labels)
    print('VAL LABELS DISTRIBUTION')
    print_class_distributions(val_labels)
    print('----------------------------')
