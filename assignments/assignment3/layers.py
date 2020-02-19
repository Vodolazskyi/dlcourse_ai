import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
        predictions, np array, shape is either (N) or (batch_size, N) -
            classifier output

    Returns:
        probs, np array of the same shape as predictions -
            probability for every class, 0..1
    '''
    preds = predictions.copy()
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    preds -= preds.max(axis=1).reshape(-1, 1)
    return np.exp(preds) / np.sum(np.exp(preds), axis=1).reshape(-1, 1)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
        probs, np array, shape is either (N) or (batch_size, N) -
            probabilities for every class
        target_index: np array of int, shape is (1) or (batch_size) -
            index of the true class for given sample(s)

    Returns:
        loss: single value
    '''
    if isinstance(target_index, int):
        target_index = np.array([target_index])
    return np.mean(
        -np.log(probs[range(len(probs)), target_index.reshape(1, -1)])
    )


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
        W, np array - weights
        reg_strength - float value

    Returns:
        loss, single value - l2 regularization loss
        gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
        predictions, np array, shape is either (N) or (batch_size, N) -
            classifier output
        target_index: np array of int, shape is (1) or (batch_size) -
            index of the true class for given sample(s)

    Returns:
        loss, single value - cross-entropy loss
        dprediction, np array same shape as predictions -
        gradient of predictions by loss value
    """
    if isinstance(target_index, int):
        target_index = np.array([target_index])
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    dprediction[range(len(probs)), target_index.reshape(1, -1)] -= 1
    dprediction /= len(probs)

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        out = np.maximum(0, X)
        self.grad = 1 * (out > 0)
        return out

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
            d_out, np array (batch_size, num_features) - gradient
                of loss function with respect to output

        Returns:
            d_result: np array (batch_size, num_features) - gradient
                with respect to input
        """
        d_result = d_out * self.grad
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return self.X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
            d_out, np array (batch_size, n_output) - gradient
                of loss function with respect to output

        Returns:
            d_result: np array (batch_size, n_input) - gradient
                with respect to input
        """
        d_input = d_out @ self.W.value.T
        self.W.grad = self.X.T @ d_out
        self.B.grad = d_out.sum(axis=0).reshape(self.B.value.shape)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        pad = self.padding
        self.X = np.zeros(
            (batch_size, height + pad*2, width + pad*2, channels)
        )
        self.X[:, pad:pad+height, pad:pad+width, :] = X.copy()

        out_height = self.X.shape[1] - self.filter_size + 1
        out_width = self.X.shape[2] - self.filter_size + 1
        output = np.zeros(
            (batch_size, out_height, out_width, self.out_channels)
        )
        f = self.filter_size
        for y in range(out_height):
            for x in range(out_width):
                X_window = self.X[:, y:y+f, x:x+f, :].reshape(batch_size, -1)
                W_r = self.W.value.reshape(-1, self.out_channels)
                output[:, y, x, :] = X_window @ W_r + self.B.value
        return output

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_input = np.zeros_like(self.X)
        f = self.filter_size

        for y in range(out_height):
            for x in range(out_width):
                X_window = self.X[:, y:y+f, x:x+f, :].reshape(batch_size, -1)
                W_r = self.W.value.reshape(-1, self.out_channels)
                d_input_window = d_out[:, y, x, :] @ W_r.T
                dW = X_window.T @ d_out[:, y, x, :]
                d_input[:, y:y+f, x:x+f, :] += \
                    d_input_window.reshape(batch_size, self.filter_size,
                                           self.filter_size, self.in_channels)
                self.W.grad += dW.reshape(self.W.value.shape)
                self.B.grad += d_out[:, y, x, :].sum(axis=0)
        if self.padding == 0:
            return d_input
        else:
            p = self.padding
            return d_input[:, p:-p, p:-p, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros((batch_size, out_height, out_width, channels))
        s = self.stride
        ps = self.pool_size
        for y in range(out_height):
            for x in range(out_width):
                out[:, y, x, :] = np.amax(X[:, y*s:y*s+ps, x*s:x*s+ps, :],
                                          axis=(1, 2))
        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_input = np.zeros_like(self.X)
        s = self.stride
        ps = self.pool_size
        for y in range(out_height):
            for x in range(out_width):
                for b in range(batch_size):
                    for c in range(channels):
                        d_out_pooled = d_out[b, y, x, c]
                        X_pooled = self.X[b, y*s:y*s+ps, x*s:x*s+ps, c]

                        max_ind_y, max_ind_x = np.unravel_index(
                            np.argmax(X_pooled), X_pooled.shape
                        )
                        d_input[b, y*s+max_ind_y, x*s+max_ind_x, c] += \
                            d_out_pooled
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
