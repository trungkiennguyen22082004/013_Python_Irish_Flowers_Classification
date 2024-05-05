import numpy as np

from dataProcessor import Data
from deepNeuralNetwork.neuralNetwork import NeuralNetwork

data = Data(datafileName="./IRIS.csv")

# data.printOutput()

# Create an instance of the Neural Network
#   Number of input features
inputDimension = data.xTrain.shape[1]  
#   Number of units in each hidden layer
hiddenDimensions = [5, 5, 5] 
#   Number of output classes
outputDimension = data.encodedYTrain.shape[1]  
#   Learning rate
learningRate = 0.01

dnn = NeuralNetwork(inputDimension=inputDimension, hiddenDimensions=hiddenDimensions, outputDimension=outputDimension, learningRate=learningRate)

numberOfEpoches = 24

for epoch in range(0, numberOfEpoches):
    # Forward propagation
    _, cost = dnn.predict(data.xTrain.T, data.encodedYTrain.T)
    
    print("\n========================================================================")
    print("Epoch [{}/{}], Cost: {}".format(epoch + 1, numberOfEpoches, cost))
        
predictedValues, testCost = dnn.predict(data.xTest.T, data.encodedYTest.T, training=False)
predictedValues = predictedValues.transpose()

print("\nTest Cost: {}".format(testCost))

decodedPredictedValues = []
for i in range(0, predictedValues.shape[0]):
    predictedValue = predictedValues[i]
    
    maxValue = max(predictedValue)
    
    if maxValue == predictedValue[0]:
        decodedPredictedValues.append("Iris-setosa") 
    elif maxValue == predictedValue[1]:
        decodedPredictedValues.append("Iris-versicolor") 
    else:
        decodedPredictedValues.append("Iris-virginicas")
        
decodedPredictedValues = np.array(decodedPredictedValues)

# print("\nActual Values:")
# print(data.yTest.shape)
# print(data.yTest)

# print("\nPredicted Values:")
# print(predictedValues.shape)
# print(predictedValues)