import keras
import keras.backend as K
import keras.layers as L
from keras.layers import Layer, InputSpec


class SliceLike:

    def __init__(self):
        self._idx = None

    def __getitem__(self, idx):
        return idx

SLICE_LIKE = SliceLike()


class Slice(Layer):
    """Slice layer

    It slice the input tensor with the provided slice tuple

    # Arguments
        slices: int tuple of ints or slices, as many as the input as dimensions

    # Input shape
        ND tensor with shape `(batch, dim1, ..., dimN, features)`

    # Output shape
        ND tensor with shape `(batch, dim1, ..., dimN, features) ^ slices`
    """

    def __init__(self, slices=None, **kwargs):
        super(Slice, self).__init__(**kwargs)
        # self.slice = conv_utils.normalize_tuple(slice, 2, 'cropping')
        assert type(slices) == tuple
        assert all(type(s) == slice for s in slices)
        # TODO elements might be int --> reduces ndim  -->  forbid or allow ???
        self.slices = slices
        self.input_spec = InputSpec(ndim=len(slices))

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == len(self.slices)

        def compute_output_slice(input_length, slc):
            if input_length is not None:
                # TODO handle negative indices
                if  slc.stop is not None:
                    length = slc.stop
                else:
                    length = input_length
                if  slc.start is not None:
                    length -= slc.start
                if  slc.step is not None:
                    length /= slc.step        # TODO round up / down?
                return length
            else:
                return None

        return tuple(
            compute_output_slice(input_length, slc)
            for input_length, slc in zip(input_shape, self.slices)
        )

    def call(self, inputs):
        return inputs[self.slices]

    def get_config(self):
        config = {'slices': self.slices}
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





if K.backend() == 'theano':

    from theano import tensor as TT

    def fft(x):
        y = TT.fft.rfft(x)
        if hasattr(x, '_keras_shape'):
            y._keras_shape = (x._keras_shape[0], x._keras_shape[1]/2 + 1, 2)
        return y

    def power_spectrum(x):
        y = TT.fft.rfft(x)
        y = y[:,:,0]**2 + y[:,:,1]**2
        # if hasattr(x, '_keras_shape'):
        #     y._keras_shape = (x._keras_shape[0], x._keras_shape[1]/2 + 1)
        return y

else:

    def fft(x):
        y = tf.fft(x)
        # ...
        return y


class FourierTrafo(Layer):
    """Fourier Transform layer

    Computes a (fast) fourier transform of the input tensor

    # Arguments
        tensor

    # Input shape
        ND tensor with shape `(batch, dim1, ..., dimN, features)`

    # Output shape
        ND tensor with shape `(batch, dim1, ..., dimN, features) ^ slices`
    """

    def __init__(self, **kwargs):
        super(FourierTrafo, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]/2 + 1, 2)

    def call(self, inputs):
        return fft(inputs)



def soft_relu(x, leak=0.0):
   return (0.5+leak) * x  +  (0.5-leak) * K.sqrt(1.0 + x*x)

def tanhx(x, alpha=0.0):
   return (1.-alpha) * K.tanh(x) + alpha * x

keras.utils.generic_utils.get_custom_objects().update({
    'soft_relu': L.Activation(soft_relu),
    'tanhx': L.Activation(tanhx)
})
