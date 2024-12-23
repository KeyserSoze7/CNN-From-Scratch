import numpy as np


"""
 Convolutional Layer

 Use: The convolutional layer focuses on feature extraction, which is critical for understanding patterns in input inages.

    Forward Pass:
        We apply convolution operations using filters to extract local patterns (e.g., edges, textures).
        we also produce a feature map that highlights important spatial features in the input.
    Backward Pass:
        Here, we compute gradients of the loss with respect to:
            Filters: Adjusts the filters to better detect features.
            Input: Allows error signals to propagate backward for earlier layers.

"""

class Conv:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels  # Number of filters applied to input
        self.kernel_size = kernel_size    # size of the filter / Kernel
        self.stride = stride   # Number of pixels the filter moves during convolution on input image
        self.padding = padding  # Adding zeros around the input to control output dimensions.
        self.filters = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1    # small matrix of weights used to extract features from the input image( By performing a convolutional operation )
        self.biases = np.zeros((output_channels, 1))

    def pad_input(self, input, pad_value=0):
        if self.padding == 0:
            return input
        return np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=pad_value)

    def forward(self, input):
        """
        Returns a 3d numpy array with dimensions (batch_size,output_channels,output_height,output_weight).

        """
        self.input = self.pad_input(input)  # Save for backward
        batch_size, in_channels, in_height, in_width = self.input.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        self.output = np.zeros((batch_size, self.output_channels, out_height, out_width))
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for i in range(0, in_height - self.kernel_size + 1, self.stride):
                    for j in range(0, in_width - self.kernel_size + 1, self.stride):
                        region = self.input[b, :, i:i+self.kernel_size, j:j+self.kernel_size]
                        self.output[b, oc, i // self.stride, j // self.stride] = np.sum(region * self.filters[oc]) + self.biases[oc]
        return self.output

    def backward(self, d_output, learning_rate=0.001):
        batch_size, _, out_height, out_width = d_output.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(self.input)

        for b in range(batch_size):
            for oc in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        region = self.input[b, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                        d_filters[oc] += d_output[b, oc, i, j] * region
                        d_input_padded[b, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += d_output[b, oc, i, j] * self.filters[oc]
                d_biases[oc] += np.sum(d_output[:, oc, :, :])

        # Removing padding from the gradient
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_input_padded

        # Update filters and biases
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases

        return d_input
