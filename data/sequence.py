import numpy as np
from keras.utils import Sequence, to_categorical

from config import NUM_CLASSES, BATCH_SIZE


class DocumentsSequence(Sequence):

    def __init__(self, documents, labels, batch_size=BATCH_SIZE,
                 num_classes=NUM_CLASSES, shuffle=True):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.documents = documents
        self.labels = labels
        self.indices = np.arange(len(self.documents))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.documents) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.get_batch_indices(idx)
        batch_documents = [self.documents[i] for i in batch_indices]
        batch_scores = [self.labels[i] for i in batch_indices]
        return self.generate_data(batch_documents, batch_scores)

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.indices)

    def get_batch_indices(self, idx):
        return self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

    def generate_data(self, documents, labels):
        documents = np.asarray(documents)
        labels = np.asarray(labels)
        return documents, to_categorical(labels, num_classes=self.num_classes)
