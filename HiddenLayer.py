from numpy import random
import numpy as np


class HiddenLayer:

    w = None
    y = None
    bias = None

    def __init__(self, feature,output, index, layers,bias_Check,Neurons_list,Activation):
        self.feature = feature
        self.output = output
        self.index = index
        self.hiddenlayers = layers
        self.bias_Check = bias_Check
        self.Neurons_list = Neurons_list
        self.Activation = Activation
        self.error_signal = []
        self.Generate_Weights_Bias(index)
        self.calc_Layer(self.feature)
        if self.Activation == 1:
            self.sigmoid()
        elif self.Activation == 2:
            self.tanh()

    def Generate_Weights_Bias(self, count):
        # get the bias
        if self.bias_Check == 1:
            self.bias = random.rand(self.Neurons_list[count],1)
        else:
            self.bias = np.zeros(self.Neurons_list[count])
            self.bias = self.bias.reshape(-1,1)
        # get the weights
        self.w = random.rand(self.Neurons_list[count],len(self.feature))

    def calc_Layer(self, feature):
        self.feature = feature
        # print("feature",self.feature)
        # print("weight",self.w)
        # print("bias",self.bias)
        self.y = np.dot(self.w, self.feature) + self.bias
        # print("y bef acti",self.y)

    def sigmoid(self):
        for i in range(len(self.y)):
            self.y[i] = 1 / (1 + np.exp(-1 * self.y[i]))
        # print("y after acti",self.y)

    def tanh(self):
        for i in range(len(self.y)):
            minus = 1 - np.exp(-1 * self.y[i])
            plus = 1 + np.exp(-1 * self.y[i])
            self.y[i] = minus / plus
        # print("y after acti",self.y)

    def sigmoid_derivative(self,err):
        err_signal = err * self.y * (1 - self.y)
        return err_signal

    def tanh_derivative(self,err):
        err_signal = err * ((1 + self.y) * (1 - self.y))
        return err_signal

    # Apply Backpropagation here
    def calculate_error_signal(self, isOutputLayer,output):
        self.error_signal.clear()
        self.output = output
        if isOutputLayer:
            # Calculating error signal for output layer
            out = None
            if self.output == 0:
                out = [[1],[0],[0]]
            elif self.output == 1:
                out = [[0],[1],[0]]
            elif self.output == 2:
                out = [[0],[0],[1]]

            signal = (out - self.y)
            if self.Activation == 1:
                signal = self.sigmoid_derivative(signal)
            elif self.Activation == 2:
                signal = self.tanh_derivative(signal)
            self.error_signal.append(signal)
        else:
            # Calculating error signal for hidden layers
            weightOfNext = self.hiddenlayers[self.index + 1].w  # Corresponding weights for the neuron
            signalOfNext = self.hiddenlayers[self.index + 1].error_signal  # the signals of next layer

            err_signal = np.dot(weightOfNext.T, signalOfNext[0])
            if self.Activation == 1:
                err_signal = self.sigmoid_derivative(err_signal)
            elif self.Activation == 2:
                err_signal = self.tanh_derivative(err_signal)
            self.error_signal.append(err_signal)

        # print("error signal ",self.error_signal)

    # Update weights
    def Update_weight(self,lr):
        weight = self.w
        for j in range(len(self.y)):
            # update bias
            if self.bias_Check == 1:
                self.bias[j] = self.bias[j] + (lr * self.error_signal[0][j])
            # update weights
            for k in range(len(self.w[j])):
                self.w[j][k] = weight[j][k] + (lr * self.error_signal[0][j]) * self.feature[k]

        # print("bias After update",self.w)
        # print("bias After update",self.bias)