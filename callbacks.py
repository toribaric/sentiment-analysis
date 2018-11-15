import numpy as np
from keras.callbacks import TensorBoard


class EnhancedTensorBoard(TensorBoard):
    def __init__(self, val_pair, vocab, batch_size, max_words, **kwargs):
        super(EnhancedTensorBoard, self).__init__(**kwargs)
        val_docs, val_labels = val_pair
        self.MAX_EMBEDDING_POINTS = 6000
        self.set_validation_data(val_docs, val_labels)
        self.set_embeddings_data(val_docs)
        self.set_embeddings_metadata(vocab, max_words, kwargs.get('log_dir'))

    def set_validation_data(self, val_docs, val_labels):
        sample_weights = np.ones(val_docs.shape[0])
        learning_phase = 0
        self.validation_data = [
            val_docs, val_labels, sample_weights, learning_phase]

    def set_embeddings_data(self, val_docs):
        embeddings_data = []
        for i, val_doc in enumerate(val_docs):
            if i == self.MAX_EMBEDDING_POINTS:
                break

            embeddings_data.append(val_doc)

        self.embeddings_data = np.array(embeddings_data)

    def set_embeddings_metadata(self, vocab, max_words, log_dir):
        metadata_path = '{}/metadata.tsv'.format(log_dir)
        self.save_embedding_metadata(vocab, metadata_path, max_words)
        self.embeddings_metadata = {'embedding_1': metadata_path}

    def save_embedding_metadata(self, vocab, metadata_path, max_words):
        with open(metadata_path, 'w') as f:
            f.write('Word\tFrequency\n')
            for row, (word, freq) in enumerate(vocab.items()):
                if freq > max_words:
                    continue

                f.write('{}\t{}{}'.format(
                    word, freq, '\n' if row < max_words else ''))
