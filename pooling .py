import numpy as np
from layer import Layer
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        strides = self.stride
        pads = self.pad
        (N, C, H, W) = inputs.shape
        inputs_reshaped = inputs.reshape(N * C, 1, H, W)
        (N_P, N_C, N_H, N_W) = inputs_reshaped.shape
        pool_H = self.pool_height
        pool_W = self.pool_width
        # calculating the height and width of next layer using formula
        inputs_reshaped_pad = np.pad(inputs_reshaped, ((0, 0), (0, 0), (pads, pads), (pads, pads)), 'constant',
                                     constant_values=0)

        out_height = 1 + int((H + 2 * pads - pool_H) / strides)
        out_width = 1 + int((W + 2 * pads - pool_W) / strides)

        i0 = np.repeat(np.arange(pool_H), pool_W)
        i0 = np.tile(i0, N_C)
        i1 = strides * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(pool_W), pool_H * N_C)
        j1 = strides * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(N_C), pool_H * pool_W).reshape(-1, 1)

        cols = inputs_reshaped_pad[:, k, i, j]
        self.X_col = cols.transpose(1, 2, 0).reshape(pool_H * pool_W * N_C, -1)

        if self.pool_type == 'max':
            self.max_ind = np.argmax(self.X_col, axis=0)
            outputs = self.X_col[self.max_ind, range(self.max_ind.size)]
        elif self.pool_type == 'avg':
            self.avg = np.mean(self.X_col, axis=0)
            outputs = self.avg

        outputs = outputs.reshape(out_height, out_width, N, C)
        outputs = outputs.transpose(2, 3, 0, 1)
        #############################################################

        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        strides = self.stride
        pads = self.pad
        pool_H = self.pool_height
        pool_W = self.pool_width
        # calculating the height and width of next layer using formula
        (N, d_C, d_H, d_W) = inputs.shape
        (N, C, H, W) = in_grads.shape

        out_grads = np.zeros(inputs.shape)

        # looping through all N examples
        for i in range(N):
            i_inputs = inputs[i, :, :, :]
            # looping through the height, width and channels
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        v_start = h * strides
                        v_end = v_start + pool_H
                        h_start = w * strides
                        h_end = h_start + pool_W

                        if self.pool_type == 'max':
                            i_out = i_inputs[c, v_start:v_end, h_start:h_end]
                            mask = (i_out == np.max(i_out))
                            out_grads[i, c, v_start:v_end, h_start:h_end] += mask * in_grads[i, c, h, w]
                        elif self.pool_type == 'avg':
                            in_grads2 = in_grads[i, c, h, w]
                            average = in_grads2 / (pool_H * pool_W)
                            out_grads[i, c, v_start:v_end, h_start:h_end] += np.ones((pool_H, pool_W)) * average
        #############################################################
        assert (out_grads.shape == inputs.shape)

        return out_grads


