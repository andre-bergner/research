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
