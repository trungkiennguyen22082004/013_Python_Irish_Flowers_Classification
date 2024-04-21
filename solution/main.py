from dataProcessor import Data
from deepNeuralNetwork.neuralNetwork import NeuralNetwork

data = Data(datafileName="IRIS.csv")

# Create an instance of the Neural Network
#   Number of input features
inputDimension = data.xTrain.shape[1]  
#   Number of units in each hidden layer
hiddenDimensions = [10, 10] 
#   Number of output classes
outputDimension = data.encodedYTrain.shape[1]  
#   Learning rate
learningRate = 0.1

dnn = NeuralNetwork(inputDimension=inputDimension, hiddenDimensions=hiddenDimensions, outputDimension=outputDimension, learningRate=learningRate)

numberOfEpoches = 10

for epoch in range(0, numberOfEpoches):
    # Forward propagation
    _, cost = dnn.predict(data.xTrain.T, data.encodedYTrain.T)
    
    print("Epoch [{}/{}], Cost: {}".format(epoch + 1, numberOfEpoches, cost))
        
predictedValues, testCost = dnn.predict(data.xTest.T, data.encodedYTest.T, training=False)
print("\nTest Cost: {}".format(testCost))
print("Predicted Values:")
print(data.xTest.T.shape)
print(data.encodedYTest.T.shape)
print(data.encodedYTest.T)
print(predictedValues.shape)
print(predictedValues)