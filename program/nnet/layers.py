
__author__ = 'Pedro Pablo'

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class ExpandedLayer(Layer):
    """
    Custom layer to be used in the expanded (Chebyshev, Legendre and Hermite)
    block. Also implements the evaluation in `tanh` as required for these
    models.
    """
    def __init__(self, output_dim, expansion, **kwargs):
        self.output_dim = output_dim
        self.expansion = expansion
        super(ExpandedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.expanded_weights = self.add_weight(name='expanded_weights',
                                                shape=(len(self.expansion),
                                                       self.output_dim),
                                                initializer='glorot_uniform',
                                                trainable=True)
        # self.tanh_weight = self.add_weight(name='tanh_weight',
        #                                   shape=(1, 1),
        #                                   initializer='glorot_uniform',
        #                                   trainable=True)
        super(ExpandedLayer, self).build(input_shape)

    def call(self, x):
        evaluation = 0
        for index, expression in enumerate(self.expansion):
            evaluation += self.expanded_weights[index] * expression(x)
        return K.tanh(evaluation)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(ExpandedLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RBFLayer(Layer):
    """
    Custom Radial Basis Functions (RBF) layer to be used in the RBF model.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim,
                                              input_shape[1]),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer='glorot_uniform',
                                     trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C = K.expand_dims(self.centers)
        H = K.transpose(C - K.transpose(x))
        p_norm = K.sum(H**2, axis=1)
        return K.exp(-self.betas**2 * p_norm)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
