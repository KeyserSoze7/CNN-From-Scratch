import numpy as np

"""
Max Pool Layer
Use: it reduces overfitting, computational cost, and ensures that small variations in the input do not significantly affect the network’s predictions.

"""

class MaxPool:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        """
        Forward Pass:
            Here we reduce the spatial dimensions of the feature maps by selecting the maximum value from each pooling region &
            Provide spatial invariance and reduces computation by downsampling the feature maps.

        """
        self.input = input
        batch_size, channels, in_height, in_width = input.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros_like(input)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, in_height - self.pool_size + 1, self.stride):
                    for j in range(0, in_width - self.pool_size + 1, self.stride):
                        region = input[b, c, i:i+self.pool_size, j:j+self.pool_size]
                        max_val = np.max(region)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        self.output[b, c, i // self.stride, j // self.stride] = max_val
                        self.max_indices[b, c, i + max_idx[0], j + max_idx[1]] = 1
        return self.output

    def backward(self, d_output):
        """
        Backward Pass:
            Here we propagate the gradient only to the positions of the maximum values in the input during the forward pass.
            No learnable parameters, so only the gradient w.r.t. the input is computed.
        """
        d_input = np.zeros_like(self.input)
        batch_size, channels, out_height, out_width = d_output.shape

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = self.max_indices[b, c, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                        d_input[b, c, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size] += d_output[b, c, i, j] * region
        return d_input
