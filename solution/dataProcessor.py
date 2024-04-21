import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Data:
    
    def __init__(self, datafileName, testProportion=0.2):
        
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
        
        print("=================================DATA===================================") 
        print(self.data)
        
        print("===================================X====================================") 
        print(self.x)
        
        print("===================================Y====================================") 
        print(self.y)
        
        print("================================X-TRAIN=================================") 
        print(self.xTrain)
        
        print("================================X-TEST==================================") 
        print(self.xTest)
        
        print("================================Y-TRAIN=================================") 
        print(self.yTrain)
        
        print("================================Y-TEST==================================") 
        print(self.yTest)
        
        print("============================ENCODED Y-TRAIN=============================") 
        print(self.encodedYTrain)
        
        print("============================ENCODED Y-TEST==============================") 
        print(self.encodedYTest)
        