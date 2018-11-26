
import keras.backend as K

if K.backend() == 'theano':

    from theano import tensor as TT

    def interleave_zeros(x, factor, axis):
        assert axis == 1
        input_shape = x.shape
        output_shape = (input_shape[0], factor*input_shape[1], input_shape[2])
        output = TT.zeros(output_shape)

        result = TT.set_subtensor(output[:, ::factor, :], x)
        if hasattr(x, '_keras_shape'):
            result._keras_shape = (x._keras_shape[0], factor*x._keras_shape[1], x._keras_shape[2])
        return result

else:

    def interleave_zeros(x, factor, axis):
        shape = K.shape(x)
        out_shape = [shape[0], shape[1], shape[2], shape[3]]
        out_shape[axis] = factor * out_shape[axis]
        out_order = [n for n in range(1,len(out_shape)+1)]
        out_order.insert(axis+1, 0)
        input_with_zeros = K.stack([x] + (factor - 1)*[K.zeros_like(x)])
        return K.reshape(K.permute_dimensions(input_with_zeros, out_order), out_shape)



from keras.engine.topology import Layer
from keras.engine import InputSpec

class UpSampling1DZeros(Layer):

    def __init__(self, upsampling_factor=2, **kwargs):
        super(UpSampling1DZeros, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        size = self.upsampling_factor * input_shape[1] if input_shape[1] is not None else None
        return (input_shape[0], size, input_shape[2])

    def call(self, inputs):
        return interleave_zeros(inputs, self.upsampling_factor, axis=1)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(UpSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class DownSampling1D(Layer):

    def __init__(self, downsampling_factor=2, **kwargs):
        super(DownSampling1D, self).__init__(**kwargs)
        self.downsampling_factor = downsampling_factor
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        size = input_shape[1] // self.downsampling_factor if input_shape[1] is not None else None
        return (input_shape[0], size, input_shape[2])

    def call(self, inputs):
        return inputs[:, ::self.downsampling_factor, :]
        # return downsample(inputs, self.downsampling_factor, axis=1)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(DownSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
