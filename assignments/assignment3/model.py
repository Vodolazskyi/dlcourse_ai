from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels,
                 conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        in_width, in_height, in_channels = input_shape
        flattener_width = in_width // (4 * 4)
        flattener_height = in_height // (4 * 4)
        self.layers = [
            ConvolutionalLayer(in_channels, conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(
                flattener_width * flattener_height * conv2_channels,
                n_output_classes
            ),
        ]

    def zero_grads(self):
        for param in self.params().values():
            param.grad[:] = 0.0

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_preds):
        d_input = d_preds
        for layer in reversed(self.layers):
            d_input = layer.backward(d_input)
        return d_input

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        self.zero_grads()
        preds = self.forward(X)
        loss, d_preds = softmax_with_cross_entropy(preds, y)
        self.backward(d_preds)
        return loss

    def predict(self, X):
        preds = self.forward(X)
        return preds.argmax(axis=1)

    def params(self):
        result = {}
        for index, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                result[f'{index}_{name}'] = param

        return result
