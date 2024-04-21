import numpy as np

# ===============================================================================================================================
# LINEAR LAYER CLASS
class Linear:
    
    def __init__(self, inputDimension, outputDimension, learningRate=3e-3):
        
        # inputDimension means the number of units in the previous hidden layer, or the input size of the input layer (if that is the previous layer)
        # outputDimension also means the number of units in the current layer
        
        self.W = np.random.rand(outputDimension, inputDimension) * 0.01
        self.b = np.zeros((outputDimension, 1))
        
        self.cache = None
        
        self.learingRate = learningRate
        
    def forward(self, A):
        
        # A is the output from the previous hidden layer with the shape of (<number of units in that previous layer>, <number of examples>)
        # A means X (matrix of examples) if the previous layer is the input layer
        
        # Cache is stored for computing the backward pass efficiently
        self.cache = (A, self.W, self.b)
        
        # Z is the result matrix after the linear transformation process, with the shape of (<number of units in the current layer>, <number of units in the previous hidden layer>)
        # The brief formula is Z = A * W + b
        
        # Calculate simply using numpy matrix multiplication:
        # Z = np.dot(self.W, A) + self.b
        # print("=========================Z - SIMPLE CALCULATING=========================")
        # print(Z)
        
        # Calculate properly:
        (previousLayerDimension, numberOfExamples) = A.shape
        
        currentLayerDimentsion = self.W.shape[0]
        
        Z = np.zeros((currentLayerDimentsion, numberOfExamples))
        
        for i in range(currentLayerDimentsion):
            for j in range(numberOfExamples):
                dot = 0
                for k in range(previousLayerDimension):
                    dot += A[k, j] * self.W[i, k]
                    
                # Add the bias
                Z[i, j] = dot + self.b[i]
                
        # Example:
        #   Number of examples: m = 4
        #   Current layer's number of units: n^[l] = 3
        #   Previous layer's number of units: n^[l-1] = 2
        #
        #   A with shape (3, 4)
        #       [ [ x_00, x_01, x_02, x_03 ],
        #         [ x_10, x_11, x_12, x_13 ],
        #         [ x_20, x_21, x_22, x_23 ] ]   
        # 
        #   W with shape (2, 3)
        #   [ [ W_00, W_01, W_02 ],
        #     [ W_10, W_11, W_12 ] ]
        # 
        #   b with shape (2, 1)
        #       [ [ b_0 ], 
        #         [ b_1] ]
        #
        #   => Z with shape (2, 4)
        #       [ [ (W_00 * x_00 + W_01 * x_10 + W_02 * + x_20 + b_0), (W_00 * x_01 + W_01 * x_11 + W_02 * + x_21 + b_0), (W_00 * x_02 + W_01 * x_12 + W_02 * + x_22 + b_0), (W_00 * x_03 + W_01 * x_13 + W_02 * + x_23 + b_0) ],
        #         [ (W_10 * x_00 + W_11 * x_10 + W_12 * + x_20 + b_1), (W_10 * x_01 + W_11 * x_11 + W_12 * + x_21 + b_1), (W_10 * x_02 + W_11 * x_12 + W_12 * + x_22 + b_1), (W_10 * x_03 + W_11 * x_13 + W_12 * + x_23 + b_1) ] ]
                
        # print("=======================Z - PROPERLY CALCULATING=========================")                 
        # print(Z)
                
        return Z
    
    def backward(self, deltaZ):
        
        # deltaZ is the gradient of the cost with respect to the output of the current linear layer
        # Alternatively, given that L is the cost function, Z is the outputs of the current linear layer
        #   deltaZ = dL / dZ
        
        # Get the previous input
        A = self.cache[0]
         
        # Get the number of examples 
        # m = A.shape[1]
        
        # Calculate the gradients:
        #   deltaW is the gradient of the cost with respect to the weights W of the current linear layer (dZ / dW)
        #       Assume that f is the cost function based on the output (L = f(Z)), g is the function to calculate the output based on W (Z = g(W))
        #       (The "chain rule")
        #       From L = f(Z) = f(g(W)), I can state that: [f(g(x))]' = f'(g(x)) * g'(x)  <=>  dL / dW = (dL / dZ) * (dZ / dW)  <=>  deltaW = deltaZ * (dZ / dW)
        #       Also, in the forward process I have implemented: Z = A * W + b. That is why: dZ / dW = d(A * W + b) / dW = A
        #       Conclude: deltaW = deltaZ * A
        deltaW = np.dot(deltaZ, A.transpose())
        
        #   deltab is the gradient of the cost with respect to the biases b of the current linear layer (dZ / db)
        #       With a similar explanation presented above, I can state that dL / db = (dL / dZ) * (dZ / db) <=> deltab = deltaZ * (dZ / db)
        #       In the forward process I have implemented: Z = A * W + b. That is why: dZ / db = d(A * W + b) / db = 1
        #       Conclude: deltab = deltaZ
        deltab = np.sum(deltaZ, axis=1)
        
        #   dA is the gradient of the cost with respect to the input of the current linear layer (which is also the output of the previous layer)
        #       This is similar to what has happened when I caculating deltaW = dL / dA = (dL / dZ) * (dZ / dA) = deltaZ * [d(A * W + b) / dA] = deltaZ * W
        deltaA = np.dot(self.W.transpose(), deltaZ)
        
        # It can be easily duduced that deltaZ, deltaW, deltab, and deltaA have the same shape as Z, W, b, and A respectively.
        # So we simply use the numpy.transpose() effectively before multiplicating the matrices to make sure that all of those required output have the precise shape. 
        # Take an example described in the forward process above: 
        #   Number of examples: m = 4
        #   Current layer's number of units: n^[l] = 3
        #   Previous layer's number of units: n^[l-1] = 2
        #   => deltaZ.shape = (2, 4), deltaW.shape = (2, 3), deltab.shape = (3, 1)
        
        # Update the weights and biases (Gradient descent)
        
        # self.W -= self.learingRate * deltaW
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i][j] -= self.learingRate * deltaW[i][j]
        for i in range(0, len(self.b)):
            self.b[i] -= self.learingRate * deltab[i]
        
        return (deltaA, deltaW, deltab)
    
# ===============================================================================================================================
# ReLU ACTIVATION FUNCTION CLASS
class ReLU:
    
    def __init__(self):
        
        self.cache = None
    
    def forward(self, A):
        
        # Cache is stored for computing the backward pass efficiently
        self.cache = (A, 1)
        
        # ReLU activation function is stated: f(x) = max(0, x)
        
        # Calculate the output of this ReLU layer
        Z = np.maximum(0, A)
        
        # print("===================================Z===================================")
        # print(Z)
        
        return Z
    
    def backward(self, deltaZ):
        
        # deltaZ is the gradient of the cost with respect to the output of this ReLU layer (dL / dZ)
        
        # Get the previous input
        A = self.cache[0]
        
        # Given that g(x) is ReLU activation function, g'(x) means:
        #   If x < 0, g'(x) = 0
        #   If x > 0, g'(x) = 1
        #   g'(x) is not defined at x = 0
        
        # deltaA is the gradient of the cost with respect to the input of this ReLU layer (dL / dA)
        # Using the chain rule, it is easy to see that: dL / dA = (dL / dZ) * (dZ / dA)
        #   Initially, calculate the gradient of the output with respect to this ReLU layer's input (dZ / dA) based on the g'(x) explanation above:
        gradientZToA = (A > 0).astype(np.float32)
        
        # print("==========================GRADIENT OF Z TO A============================")     
        # print(gradientZToA)
        
        #   Then, calculate deltaA:
        deltaA = deltaZ * gradientZToA
        
        # print("===============================DELTA-A==================================")     
        # print(deltaA)
        
        return deltaA     
    
# ===============================================================================================================================
# SIGMOID ACTIVATION FUNCTION CLASS
class Sigmoid:
    
    def __init__(self):
        self.cache = None
        
    def forward(self, A):
        
        # Cache is stored for computing the backward pass efficiently
        self.cache = (A, 1)
        
        Z = 1 / (1 + np.exp(-A))
        
        return Z
    
    def backward(self, deltaZ):
        
        # Get the previous input
        A = self.cache
        
        # Calculate the gradient of the output with the respect to this Sigmoid layer's input (dZ / dA):
        gradientZToA = np.exp(-A) / ((1 + np.exp(-A)) ** 2)
        
        # deltaA is the gradient of the cost with respect to the input of this ReLU layer (dL / dA)
        # Using the chain rule, it is easy to see that: dL / dA = (dL / dZ) * (dZ / dA) = deltaZ * gradientZToA
        deltaA = deltaZ * gradientZToA
        
        return deltaA
    
# ===============================================================================================================================
# SOFTMAX ACTIVATION FUNCTION CLASS
class Softmax:
    
    def __init__(self):
        self.cache = None
        
    def forward(self, A):
        # Cache is stored for computing the backward pass efficiently
        self.cache = A
        
        # Compute the exponential values
        expA = np.exp(A - np.max(A, axis=0, keepdims=True))
        
        # Compute the sum of exponential values for each example
        expASum = np.sum(expA, axis=0, keepdims=True)
        
        # Compute the softmax output
        Z = expA / expASum
        
        return Z
    
    def backward(self, deltaZ):
        # Get the cached input
        A = self.cache
        
        # Compute the softmax output again
        expA = np.exp(A - np.max(A, axis=0, keepdims=True))
        expASum = np.sum(expA, axis=0, keepdims=True)
        softmax = expA / expASum
        
        # Compute the gradient of the softmax function
        gradient = softmax * (1 - softmax)
        
        # Compute the gradient of the cost with respect to the input of the softmax layer
        deltaA = deltaZ * gradient
        
        return deltaA

# ===============================================================================================================================
# CLASS FOR LAYER THAT COMBINES LINEAR LAYER AND ACTIVATION-FUNCTION LAYER
class StandardLayer:
    
    def __init__(self, inputDimension, outputDimension, learningRate, activationFunction):
        
        self.linearLayer = Linear(inputDimension, outputDimension, learningRate)
        
        self.activationFunctionLayer = None
        if activationFunction == "ReLU":
            self.activationFunctionLayer = ReLU()
        elif activationFunction == "Sigmoid":
            self.activationFunctionLayer = Sigmoid()
        elif activationFunction == "Softmax":
            self.activationFunctionLayer = Softmax()
        else:
            self.activationFunctionLayer = ReLU()    
            
    def forward(self, A):
        
        Z = self.linearLayer.forward(A)
        Z = self.activationFunctionLayer.forward(Z)
        
        return Z      
    
    def backward(self, deltaZ):
        
        # deltaZ is the gradient of the cost with the respect to the post-activation output
        # deltaHidden is the gradient of the cost with the respect to the post-linear function output
        
        deltaHidden = self.activationFunctionLayer.backward(deltaZ)
        (deltaA, deltaW, deltab) = self.linearLayer.backward(deltaZ)
        
        return (deltaA, deltaW, deltab)

# ===============================================================================================================================
# LINEAR TEST
# linear = Linear(3, 2)

# A = np.random.rand(3, 4) * 0.01

# print("===================================A====================================") 
# print(A)
# print("===================================W====================================") 
# print(linear.W)

# linear.forward(A)

# ===============================================================================================================================
# ReLU TEST
# reLU = ReLU()

# A = np.random.rand(3, 4)
# deltaZ = np.random.randn(3, 4)

# print("===================================A====================================") 
# print(A)
# print("================================DELTA-Z=================================")
# print(deltaZ)

# reLU.forward(A)
# reLU.backward(deltaZ)

# ===============================================================================================================================
# Softmax TEST
# softmax = Softmax()

# A = np.random.rand(3, 4)
# A = np.random.randint(0, 9, (3, 4))
# deltaZ = np.random.rand(3, 4)
# deltaZ = np.random.randint(0, 9, (3, 4))

# print("===================================A====================================") 
# print(A)
# print("================================DELTA-Z=================================")
# print(deltaZ)

# softmax.forward(A)
# softmax.backward(deltaZ)