import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Data:
    
    def __init__(self, datafileName, testProportion=0.5):
        
        # Load the dataset
        self.data = pd.read_csv(filepath_or_buffer="dataset/{}".format(datafileName))
        
        # Extract features and target variable
        self.x = self.data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        self.y = self.data['species'].values
        
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x, self.y, test_size=testProportion)
        
        # With one-hot encoding, each output will be represented as a binary vector:
        #   Iris-setosa: [1, 0, 0]
        #   Iris-versicolor: [0, 1, 0]
        #   Iris-virginica: [0, 0, 1]
        encoder = OneHotEncoder(sparse_output=False)
        self.encodedYTrain = encoder.fit_transform(self.yTrain.reshape(-1, 1))
        self.encodedYTest = encoder.transform(self.yTest.reshape(-1, 1))
        
    def printOutput(self):
        
        print("\n=================================DATA===================================") 
        print("Data's shape: {}".format(self.data.shape))
        print(self.data)
        
        print("\n===================================X====================================")
        print("X's shape: {}".format(self.x.shape))
        print(self.x)
        
        print("\n===================================Y====================================") 
        print("Y's shape: {}".format(self.y.shape))
        print(self.y)
        
        print("\n================================X-TRAIN=================================") 
        print("X-Train's shape: {}".format(self.xTrain.shape))
        print(self.xTrain)
        
        print("\n================================X-TEST==================================") 
        print("X-Test's shape: {}".format(self.xTest.shape))
        print(self.xTest)
        
        print("\n================================Y-TRAIN=================================") 
        print("Y-Train's shape: {}".format(self.yTrain.shape))
        print(self.yTrain)
        
        print("\n================================Y-TEST==================================") 
        print("Y-Test's shape: {}".format(self.yTest.shape))
        print(self.yTest)
        
        print("\n============================ENCODED Y-TRAIN=============================") 
        print("Encoded Y-Train's shape: {}".format(self.encodedYTrain.shape))
        print(self.encodedYTrain)
        
        print("\n============================ENCODED Y-TEST==============================") 
        print("Encoded Y-Test's shape: {}".format(self.encodedYTest.shape))
        print(self.encodedYTest)
        
        print()
        