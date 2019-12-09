import numpy as np

neuron = 4


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        print('inputs \n', self.input)
        print()
        self.weights1 = np.random.rand(self.input.shape[1], neuron)
        print('weights1 \n', self.weights1)
        print()
        self.weights2 = np.random.rand(neuron, 1)
        print('weights2 \n', self.weights2)
        print()
        self.y = y
        print('y \n', self.y)
        print()
        self.output = np.zeros(self.y.shape)  # y hat
        print('output \n', self.output)
        print()

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        #        print('layer 1 \n',self.layer1)
        #        print()
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    #        print('output \n',self.output)
    #        print()

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        #        print('d_weights2  \n',d_weights2  )
        #        print()
        d_weights1 = np.dot(self.input.T,
                            (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                    self.weights2.T) * sigmoid_derivative(self.layer1)))
        #        print('d_weights1 \n',d_weights1)
        #        print()

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

nn = NeuralNetwork(X, y)

for i in range(20000):
    nn.feedforward()
    nn.backprop()
#    print('--------------------------------')
#
print(nn.output)
