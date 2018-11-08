from keras import backend as K
from keras.engine.topology import Layer


class Attention(Layer):
    """
    http://www.aclweb.org/anthology/P16-2034
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim,),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)

    def calculate_alpha(self, x):
        M = K.tanh(x)
        M = K.reshape(M, [-1, K.shape(x)[-1]])
        self.kernel = K.expand_dims(self.kernel, -1)
        text = K.dot(M, self.kernel)
        text = K.reshape(text, [-1, K.shape(x)[1]])
        return K.softmax(text)

    def calculate_output(self, x, alpha):
        x = K.permute_dimensions(x, (0, 2, 1))
        alpha = K.expand_dims(alpha, -1)
        r = K.batch_dot(x, alpha)
        r = K.squeeze(r, -1)
        return K.tanh(r)

    def call(self, x):
        alpha = self.calculate_alpha(x)
        return self.calculate_output(x, alpha)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
