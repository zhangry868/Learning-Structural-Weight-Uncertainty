import numpy as np
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer


__all__ = [
    "DenseLayer",
    "NINLayer",
    "NonlinearityLayer"
]

def HFtransform(input, vecotrMatrix, dim, K):
    orthoM = T.eye(dim) - 2 * T.outer(vecotrMatrix[:,0], vecotrMatrix[:,0]) / T.sum(vecotrMatrix[:,0]**2)
    output = T.dot(input, orthoM)
    for i in range(1,K):
        temp = T.eye(dim) - 2 * T.outer(vecotrMatrix[:,i], vecotrMatrix[:,i]) / T.sum(vecotrMatrix[:,i]**2)
        output = T.dot(output, temp)
    return output

class DenseLayer(Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should  be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        v11 = init.Constant(1)
        v12 = init.Constant(1)
        v21 = init.Constant(1)
        v22 = init.Constant(1)
        v13 = init.Constant(1)
        v23 = init.Constant(1)
        v14 = init.Constant(1)
        v24 = init.Constant(1)

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        self.v11 = self.add_param(v11, (num_inputs, ), name='v11')
        self.v21 = self.add_param(v21, (num_units, ), name='v21')
        self.v12 = self.add_param(v12, (num_inputs, ), name='v12')
        self.v22 = self.add_param(v22, (num_units, ), name='v22')
        self.v13 = self.add_param(v13, (num_inputs, ), name='v13')
        self.v23 = self.add_param(v23, (num_units, ), name='v23')
        self.v14 = self.add_param(v14, (num_inputs, ), name='v14')
        self.v24 = self.add_param(v24, (num_units, ), name='v24')

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        #input = HFtransform(input, self.v1, self.num_inputs, self.K)
        self.p = T.eye(self.num_inputs) - 2 * T.outer(self.v11, self.v11)/T.sum(self.v11**2)
        input = T.dot(input, self.p) 
        self.p = T.eye(self.num_inputs) - 2 * T.outer(self.v12, self.v12)/T.sum(self.v12**2)
        input = T.dot(input, self.p) 
        self.p = T.eye(self.num_inputs) - 2 * T.outer(self.v13, self.v13)/T.sum(self.v13**2)
        input = T.dot(input, self.p) 
        self.p = T.eye(self.num_inputs) - 2 * T.outer(self.v14, self.v14)/T.sum(self.v14**2)
        input = T.dot(input, self.p)
        input = T.dot(input, self.W)        
        self.q = T.eye(self.num_units) - 2 * T.outer(self.v21, self.v21)/T.sum(self.v21**2)
        input = T.dot(input, self.q)
        self.q = T.eye(self.num_units) - 2 * T.outer(self.v23, self.v23)/T.sum(self.v22**2)
        input = T.dot(input, self.q)
        self.q = T.eye(self.num_units) - 2 * T.outer(self.v23, self.v23)/T.sum(self.v23**2)
        input = T.dot(input, self.q)
        self.q = T.eye(self.num_units) - 2 * T.outer(self.v24, self.v24)/T.sum(self.v24**2)
        input = T.dot(input, self.q)
        activation = input
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class NonlinearityLayer(Layer):
    """
    lasagne.layers.NonlinearityLayer(incoming,
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A layer that just applies a nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    """
    def __init__(self, incoming, nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(NonlinearityLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(input)


class NINLayer(Layer):
    """
    lasagne.layers.NINLayer(incoming, num_units, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    Network-in-network layer.
    Like DenseLayer, but broadcasting across all trailing dimensions beyond the
    2nd.  This results in a convolution operation with filter size 1 on all
    trailing dimensions.  Any number of trailing dimensions is supported,
    so NINLayer can be used to implement 1D, 2D, 3D, ... convolutions.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    untie_biases : bool
        If false the network has a single bias vector similar to a dense
        layer. If true a separate bias vector is used for each trailing
        dimension beyond the 2nd.

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should be (num_inputs, num_units),
        where num_units is the size of the 2nd. dimension of the input.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the correct shape is determined by the
        untie_biases setting. If untie_biases is False, then the shape should
        be (num_units, ). If untie_biases is True then the shape should be
        (num_units, input_dim[2], ..., input_dim[-1]). If None is provided the
        layer will have no biases.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, NINLayer
    >>> l_in = InputLayer((100, 20, 10, 3))
    >>> l1 = NINLayer(l_in, num_units=5)

    References
    ----------
    .. [1] Lin, Min, Qiang Chen, and Shuicheng Yan (2013):
           Network in network. arXiv preprint arXiv:1312.4400.
    """
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[1]

        self.W = self.add_param(W, (num_input_channels, num_units), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_units,) + self.output_shape[2:]
            else:
                biases_shape = (num_units,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units) + input_shape[2:]

    def get_output_for(self, input, **kwargs):
        # cf * bc01... = fb01...
        out_r = T.tensordot(self.W, input, axes=[[0], [1]])
        # input dims to broadcast over
        remaining_dims = range(2, input.ndim)
        # bf01...
        out = out_r.dimshuffle(1, 0, *remaining_dims)

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                # no broadcast
                remaining_dims_biases = range(1, input.ndim - 1)
            else:
                remaining_dims_biases = ['x'] * (input.ndim - 2)  # broadcast
            b_shuffled = self.b.dimshuffle('x', 0, *remaining_dims_biases)
            activation = out + b_shuffled

        return self.nonlinearity(activation)
