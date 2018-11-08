import os
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    Embedding, LSTM, CuDNNLSTM, Dense, SpatialDropout1D, Dropout)
from keras.callbacks import TensorBoard, ModelCheckpoint

from layers import Attention
from config import MAX_LEN, NUM_CLASSES


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
        model.add(self.LSTM(288, return_sequences=True))
        model.add(Attention(288))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()
        return model

    def train(self, training_sequence, validation_sequence):
        self.keras_model.fit_generator(
            epochs=10,
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

        sentiments = ['negative', 'neutral', 'positive']
        return sentiments[np.argmax(predictions)]

    def load_weights(self, weights_path):
        self.keras_model.load_weights(weights_path)

    def callbacks(self):
        log_dir = self.config.get('log_dir')
        checkpoint_path = os.path.join(log_dir, 'weights_{epoch:04d}.h5')
        return [
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(
                checkpoint_path, verbose=0, save_weights_only=True),
        ]
