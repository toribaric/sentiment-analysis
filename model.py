import os
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    Embedding, LSTM, CuDNNLSTM, Dense, SpatialDropout1D, Dropout,
    Bidirectional)
from keras.callbacks import ModelCheckpoint

from layers import Attention
from callbacks import EnhancedTensorBoard
from config import MAX_LEN, MAX_WORDS, NUM_CLASSES, SENTIMENTS, BATCH_SIZE


class SentimentModel(object):

    def __init__(self, config={}):
        num_gpus = len(K.tensorflow_backend._get_available_gpus())
        self.config = config
        self.LSTM = CuDNNLSTM if num_gpus > 0 else LSTM
        self.keras_model = self.build()

    def build(self):
        model = Sequential()
        model.add(Embedding(self.config.get('vocab_size'),
                            128, input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.7))
        model.add(Bidirectional(self.LSTM(192, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Attention(384))
        model.add(Dropout(0.2))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()
        return model

    def train(self, training_sequence, validation_sequence):
        self.keras_model.fit_generator(
            epochs=21,
            generator=training_sequence,
            validation_data=validation_sequence,
            shuffle=True,
            use_multiprocessing=True,
            workers=12,
            callbacks=self.callbacks()
        )

    def analyze(self, sequence, verbose=0):
        predictions = self.keras_model.predict(sequence, verbose=verbose)
        if verbose:
            print(predictions)

        return SENTIMENTS[np.argmax(predictions)]

    def load_weights(self, weights_path):
        self.keras_model.load_weights(weights_path)

    def callbacks(self):
        log_dir = self.config.get('log_dir')
        checkpoint_path = os.path.join(log_dir, 'weights_{epoch:04d}.h5')
        tensorboard_config = self.config.get('tensorboard')
        return [
            EnhancedTensorBoard(
                val_pair=tensorboard_config.get('val_pair'),
                vocab=tensorboard_config.get('vocab'),
                batch_size=BATCH_SIZE,
                max_words=MAX_WORDS,
                log_dir=log_dir,
                histogram_freq=1,
                embeddings_freq=10,
            ),
            ModelCheckpoint(
                checkpoint_path, verbose=0, save_weights_only=True),
        ]
