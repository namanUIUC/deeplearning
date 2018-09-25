"Model"

import numpy as np

#from model.utils.layers import Layer
#from model.utils.utility_methods import softmax
#from model.utils.utility_methods import mean_cross_entropy_softmax
#from model.utils.utility_methods import d_mean_cross_entropy_softmax


class Perceptron(object):
    "Perceptron Model"

    def __init__(self, input_dim, output_dim, hidden_dims, activ_funcs):
        """
        Input:
          input_dim     : dimension of input (int)
          output_dim    : dimension of output( int)
          hidden_dims    : a list of integers specifying the number of
                            hidden units on each layer.
          activ_funcs   : a list of function objects specifying
                            the activation function of each layer.
        """

        self.units = [input_dim] + hidden_dims[:] + [output_dim]
        self.activ_funcs = activ_funcs[:] + [linear]

        self.shapes = []
        self.layers = []

        self.logits = None
        self.pred_prob = None

        self._build_network(self.units)

    def _build_network(self, neuron_list):
        """
        Build the network by assigning tuples to shapes and
        objects to layers.
        Input:
          neuron_list     : list of units in each layers (list)
        """

        # num_input and num_outputs assignment per layer
        for i in range(len(neuron_list) - 1):
            self.shapes.append((neuron_list[i], neuron_list[i + 1]))

        # creating layers
        for i, shape in enumerate(self.shapes):
            self.layers.append(Layer(shape, self.activ_funcs[i]))

    def loss(self, outputs, gt):
        """
        Compute the cross entropy softmax loss

        Inputs:
            outputs : output of the last layer
            gt      : ground truth lables

        Returns:
            mean_cross_entropy_softmax
        """
        return mean_cross_entropy_softmax(outputs, gt)

    def d_loss(self, outputs, gt):
        """
        Compute derivatives of the cross entropy softmax loss w.r.t the outputs.

        Inputs:
            outputs : output of the last layer
            gt      : ground truth lables

        Returns:
            derivative w.r.t mean_cross_entropy_softmax
        """
        return d_mean_cross_entropy_softmax(outputs, gt)

    def forward(self, x, gt=None):
        """
        Network inference / forward propogation.

        Inputs:
            x  : features of dim[batch_size, feature_dims] (np.array)
            gt : lables of dim[batch_size, lable_dims] (np.array) [optional]

        Returns:
            pred_prob : prediction probability of dim[batch_size, lable_dims] (np.array)
            loss      : loss value (float) [if gt is None then None]
        """
        layer_inputs = x
        for layer in self.layers:
            layer_outputs = layer.forward(layer_inputs)
            layer_inputs = layer_outputs

        self.logits = layer_outputs

        self.pred_prob = softmax(self.logits)

        if gt is None:
            return self.pred_prob, None
        else:
            return self.pred_prob, self.loss(layer_outputs, gt)

    def backward(self, gt):
        """
        Network train / back propogation.

        Inputs:
            gt : lables of dim[batch_size, lable_dims] (np.array) [optional]

        """
        d_layer_outputs = self.d_loss(self.layers[-1].a, gt)
        for layer in self.layers[::-1]:
            d_layer_inputs = layer.backward(d_layer_outputs)
            d_layer_outputs = d_layer_inputs

    def predict(self, x):
        '''
        Network predition

        Inputs:
            x  : features of dim[batch_size, feature_dims] (np.array)

        Outputs:
            y_predict : one hot predition of dim[batch_size, lable_dims] (np.array)
        '''
        pred_prob, _ = self.forward(x)
        return np.argmax(pred_prob, axis=1)


class Layer(object):
    "Implements a layer of a NN."

    def __init__(self, shape, activ_func):

        # Weight matrix of dims[L-1, L]
        self.w = np.random.uniform(-np.sqrt(2.0 / shape[0]),
                                   np.sqrt(2.0 / shape[0]),
                                   size=shape)

        # Bias marix of dims[1, L]
        self.b = np.zeros((1, shape[1]))

        # The activation function
        self.activate = activ_func

        # The derivative of the activation function.
        self.d_activate = d_fun[activ_func]

    def forward(self, inputs):
        '''
        Forward propagate through this layer.

        Inputs:
            inputs : inputs to this layer of dim[N, L-1] (np.array)
        Outputs:
            outputs: output of this layer of dim[N, L] (np.array)

        '''
        # cache for backward
        self.inputs = inputs

        # Linear score
        score = np.dot(inputs, self.w) + self.b

        # Activation
        outputs = self.activate(score)

        # cache for backward
        self.a = outputs

        return outputs

    def backward(self, d_outputs):
        """
        Backward propagate the gradient through this layer.

        Inputs:
            d_outputs : deltas from the previous(deeper) layer of dims[N, L] (np.array)

        Outputs:
            d_inputs : deltas for this layer of dims[N, L-1] (np.array)

        """

        # Derivatives of the loss w.r.t the scores (the result from linear transformation).
        d_scores = d_outputs * self.d_activate(self.a)

        # Derivatives of the loss w.r.t the bias, averaged over all data points.
        self.d_b = np.mean(d_scores, axis=0, keepdims=True)

        # Derivatives of the loss w.r.t the weight matrix, averaged over all data points.
        self.d_w = np.dot(self.inputs.T, d_scores) / d_scores.shape[0]

        # Derivatives of the loss w.r.t the previous layer's activations/outputs.
        d_inputs = np.dot(d_scores, self.w.T)

        return d_inputs


class GradientDescentOptimizer(object):
    "Gradient descent with staircase exponential decay."

    def __init__(self, learning_rate, decay_steps=1000,
                 decay_rate=1.0):

        self.learning_rate = learning_rate
        self.steps = 0.0
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def update(self, model):
        '''
        Update model parameters.

        Inputs:
            model : Model (model obj)
        '''
        for layer in model.layers:
            layer.w -= layer.d_w * self.learning_rate
            layer.b -= layer.d_b * self.learning_rate
        self.steps += 1
        if (self.steps + 1) % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate


# Utility functions.

def linear(x):
    "linear function"
    return x


def d_linear(a=None, x=None):
    "Derivative of linear function"
    return 1.0


def sigmoid(x):
    "The sigmoid function."
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    "Derivative of sigmoid function"
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    "The rectified linear activation function."
    return np.clip(x, 0.0, None)


def d_relu(a=None, x=None):
    "Derivative of RELU given activation (a) or input (x)."
    if a is not None:
        d = np.zeros_like(a)
        d[np.where(a > 0.0)] = 1.0
        return d
    else:
        return d_relu(a=relu(x))


def tanh(x):
    "The tanh activation function."
    return np.tanh(x)


def d_tanh(a=None, x=None):
    "The derivative of the tanh function."
    if a is not None:
        return 1 - a ** 2
    else:
        return d_tanh(a=tanh(x))


def softmax(x):
    "Softmax function"
    # For numerical stability mentioned in CS226 UCB
    shifted_x = x - np.max(x, axis=1, keepdims=True)

    f = np.exp(shifted_x)
    p = f / np.sum(f, axis=1, keepdims=True)
    return p


def mean_cross_entropy(p, y):
    "Mean cross entropy"
    n = y.shape[0]
    return - np.sum(y * np.log(p)) / n


def mean_cross_entropy_softmax(logits, y):
    "Mean cross entropy with the softmax function"
    return mean_cross_entropy(softmax(logits), y)


def d_mean_cross_entropy_softmax(logits, y):
    "derivative of the Error w.r.t Mean cross entropy with the softmax function"
    return softmax(logits) - y


# Mapping from activation functions to its derivatives.
d_fun = {relu: d_relu,
         tanh: d_tanh,
         sigmoid: d_sigmoid,
         linear: d_linear}
