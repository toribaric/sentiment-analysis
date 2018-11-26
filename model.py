import os
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import (
    Input, Embedding, LSTM, CuDNNLSTM, Dense, SpatialDropout1D, Dropout,
    Bidirectional)
from keras.callbacks import ModelCheckpoint

from layers import Attention
from callbacks import EnhancedTensorBoard
from config import (
    MAX_LEN, MAX_WORDS, NUM_SENTIMENTS, NUM_CATEGORIES,
    PREDICTED_SENTIMENT_VALUES, PREDICTED_CATEGORY_VALUES, BATCH_SIZE)


class SentimentModel(object):

    def __init__(self, config={}):
        num_gpus = len(K.tensorflow_backend._get_available_gpus())
        self.config = config
        self.LSTM = CuDNNLSTM if num_gpus > 0 else LSTM
        self.keras_model = self.build()

    def build(self):
        inputs = Input(shape=(MAX_LEN,), name='input')
        lstm_output = self.build_lstm(inputs)
        sen_output = self.build_sentiment_classifier(lstm_output)
        cat_output = self.build_category_classifier(lstm_output)
        model = Model(inputs=inputs, outputs=[sen_output, cat_output])
        model.compile(optimizer='adam', metrics=['acc'], **self.loss_params())
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

        return (
            PREDICTED_SENTIMENT_VALUES[np.argmax(predictions[0])],
            PREDICTED_CATEGORY_VALUES[np.argmax(predictions[1])]
        )

    def load_weights(self, weights_path):
        self.keras_model.load_weights(weights_path)

    # Private methods

    def build_lstm(self, inputs):
        x = Embedding(
            self.config.get('vocab_size'), 128, input_length=MAX_LEN)(inputs)
        x = SpatialDropout1D(0.7)(x)
        x = Bidirectional(self.LSTM(192, return_sequences=True))(x)
        x = Dropout(0.5)(x)
        return Bidirectional(self.LSTM(192, return_sequences=True))(x)

    def build_sentiment_classifier(self, x):
        x = Attention(384)(x)
        x = Dropout(0.2)(x)
        return Dense(
            NUM_SENTIMENTS, activation='softmax', name='sen_output')(x)

    def build_category_classifier(self, x):
        x = Attention(384)(x)
        x = Dropout(0.2)(x)
        return Dense(
            NUM_CATEGORIES, activation='softmax', name='cat_output')(x)

    def loss_params(self):
        return {
            'loss': {
                'sen_output': 'categorical_crossentropy',
                'cat_output': 'categorical_crossentropy'
            },
            'loss_weights': {'sen_output': 1.0, 'cat_output': 0.2},
        }

    def callbacks(self):
        log_dir = self.config.get('log_dir')
        checkpoint_path = os.path.join(log_dir, 'weights_{epoch:04d}.h5')
        tensorboard_config = self.config.get('tensorboard')
        return [
            EnhancedTensorBoard(
                val_tuple=tensorboard_config.get('val_tuple'),
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
