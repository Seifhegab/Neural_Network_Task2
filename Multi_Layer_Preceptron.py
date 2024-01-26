import Evaluation
from HiddenLayer import HiddenLayer


class MultiLayerPerceptron:
    bias = None
    w = None
    input_layer = None
    y = None
    feature = None
    output = None
    y_list = []
    layers = []

    def __init__(self, x_train, y_train, x_test, y_test, lr, bias_Check, epochs,Neurons_list,numOfLayers,Activation):
        self.x_train = x_train
        self.y_train = y_train
        self.lr = lr
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.bias_Check = bias_Check
        self.Neurons_list = Neurons_list
        self.numOfLayers = numOfLayers
        self.Activation = Activation

    def generateLayers(self):
        # Forward Propagation
        self.feature = self.x_train[0].reshape(5, 1)
        self.output = self.y_train[0]
        for i in range(self.numOfLayers):
            if i == 0:
                first_layer = HiddenLayer(self.feature, self.output, i, self.layers, self.bias_Check,
                                          self.Neurons_list, self.Activation)

                self.layers.append(first_layer)
            else:
                layer = HiddenLayer(self.layers[i - 1].y, self.output, i, self.layers, self.bias_Check,
                                    self.Neurons_list, self.Activation)
                self.layers.append(layer)

        # Backpropagation Testing
        for i in range(self.numOfLayers - 1, -1, -1):
            # print("Layer", i)
            self.layers[i].calculate_error_signal(i + 1 == self.numOfLayers,self.output)

        # Forward Propagation
        for i in range(self.numOfLayers):
            self.layers[i].Update_weight(self.lr)

        # number of epochs
        for e in range(self.epochs):

            if e == 0:
                # next iterations
                k = 1
            else:
                # epochs
                k = 0
            for j in range(k, len(self.x_train)):
                # Forward Propagation
                self.feature = self.x_train[j].reshape(5, 1)
                self.output = self.y_train[j]
                for i in range(self.numOfLayers):
                    if i == 0:
                        self.layers[i].calc_Layer(self.feature)
                        if self.Activation == 1:
                            self.layers[i].sigmoid()
                        elif self.Activation == 2:
                            self.layers[i].tanh()
                    else:
                        self.layers[i].calc_Layer(self.layers[i - 1].y)
                        if self.Activation == 1:
                            self.layers[i].sigmoid()
                        elif self.Activation == 2:
                            self.layers[i].tanh()

                # Backpropagation Testing
                for i in range(self.numOfLayers - 1, -1, -1):
                    self.layers[i].calculate_error_signal(i + 1 == self.numOfLayers, self.output)

                # Forward Propagation
                for i in range(self.numOfLayers):
                    self.layers[i].Update_weight(self.lr)

        # testing
        y = []
        for j in range(0,len(self.x_test)):
            # Forward Propagation
            self.feature = self.x_test[j].reshape(5, 1)
            for i in range(self.numOfLayers):
                if i == 0:
                    self.layers[i].calc_Layer(self.feature)
                    if self.Activation == 1:
                        self.layers[i].sigmoid()
                    elif self.Activation == 2:
                        self.layers[i].tanh()
                else:
                    self.layers[i].calc_Layer(self.layers[i - 1].y)
                    if self.Activation == 1:
                        self.layers[i].sigmoid()
                    elif self.Activation == 2:
                        self.layers[i].tanh()

            maximum = max(self.layers[self.numOfLayers - 1].y)
            y_pred = self.layers[self.numOfLayers - 1].y
            for l in range(len(y_pred)):
                if y_pred[l] == maximum:
                    y.append(l)

        self.y_test = self.y_test.reshape(1,len(self.y_test))
        self.y_test = self.y_test[0]
        # print(y)
        # print(self.y_test)
        Evaluation.calculate_accuracy(y,self.y_test)
        Evaluation.confusion_matrix(y,self.y_test)