import pickle

import numpy as np

from deepNeuralNetwork.layers import StandardLayer, Linear

class NeuralNetwork:
    
    def __init__(self, inputDimension, hiddenDimensions, outputDimension, learningRate, activationFunctions=None):
        
        self.dimensions = [inputDimension] + hiddenDimensions
        
        print("=========")
        print(self.dimensions)
        
        if activationFunctions is None:
            activationFunctions = ["Softmax"] * (len(hiddenDimensions) + 1)
        elif len(activationFunctions) < hiddenDimensions:
            activationFunctions += ["Softmax"] * (len(hiddenDimensions) - len(activationFunctions) + 1)
        
        # Initialize the hidden layers
        self.layers = []
        for i in range(0, len(hiddenDimensions)):
            self.layers.append(StandardLayer(inputDimension=self.dimensions[i], outputDimension=self.dimensions[i + 1], learningRate=learningRate, activationFunction=activationFunctions[i]))
            
        # Initialize the final layer
        self.finalLayer = StandardLayer(inputDimension=self.dimensions[-1], outputDimension=outputDimension, learningRate=learningRate, activationFunction=activationFunctions[-1])
        
    def updateLearningRate(self, value):
        
        for layer in self.layers:
            layer.linear.learningRate = value
            
        self.finalLayer.learningRate = value
    
    def predict(self, A, Y, training=True):
        
        # Y is 3the actual label
        
        hidden = A
        
        # Forward pass through the layers
        for layer in self.layers:
            hidden = layer.forward(hidden)
            
        # AL is the output of the forward propagation
        AL = self.finalLayer.forward(hidden)
        
        # Calculate the cost
        cost = self.computeCost(AL, Y)
        
        if training:
            # If the process is training, the we apply the backpropagation:
            #   Calculate the gradient of the cost with respect to the scores
            # deltaAL = self.computeCostGradient(AL, Y)
            
            #   Backward pass to the final layer
            (deltaAL, _, _) = self.finalLayer.backward(AL)
            
            #   Backward pass to the remaining layers
            for i in range(len(self.layers) - 1, -1, -1):
                (deltaAL, _, _) = self.layers[i].backward(deltaAL)
            
        return AL, cost
        
    def computeCost(self, AL, Y):
        
        # Y the actual label. Y.shape = (1, m)
        # Get the number of examples
        m = Y.shape[1]
        
        # Calculate the cross-entropy cost
        cost = (-1 / m) * (np.dot(Y, (np.log(AL)).transpose()) + np.dot((1 - Y), (np.log(1 - AL)).transpose()))
        
        cost = np.squeeze(cost)
        
        return cost
    
    def computeCostGradient(self, AL, Y):
        
        # Get the number of examples
        m = Y.shape[1]
        
        # Cost formula:
        # J = (-1 / m) * [(y_1 * log(a_1) + (1 - y_1) * log (1 - a_1)) + (y_2 * log(a_2) + (1 - y_2) * log (1 - a_2)) + ... + (y_m * log(a_m) + (1 - y_m) * log (1 - a_m))]
        # So, foreach pair of (a_i, y_i), the gradient of the cost with respect to a_i is that:                             
        # J' =  (-1 / m) * [(const)' + ((y_i * log(a_i) + (1 - y_i) * log (1 - a_i)))' ]
        #    =  (-1 / m) * [0 + (y / a) + (-(1 - y) / (1 - a))]     
        #    =  (-1 / m) * (y - a) / [a * (1 - a)]
        
        # deltaAL is the gradient of the cost with respect to AL
        deltaAL = (-1 / m) * (np.divide(Y, AL.transpose()) - np.divide(1 - Y, 1 - AL))
        # deltaAL = (-1 / m) * ((Y / AL) - ((1 - Y) / (1 - AL)))
        
        return deltaAL