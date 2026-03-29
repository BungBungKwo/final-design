import numpy

class ELMRegressor:
    def __init__(self, HiddenLayerNeuronNum=85, ActiveFunc="Sigmoid", RandomState=None):
        # Arguments
        self.HiddenLayerNeuronNum = HiddenLayerNeuronNum
        self.ActiveFunc = ActiveFunc
        self.RandomState = RandomState
        
    def _ActiveFunc(self, X):
        if self.ActiveFunc == "Sigmoid":
            return (1/(1+numpy.exp(-X)))
        elif self.ActiveFunc == "Tanh":
            return numpy.tanh(X)
        elif self.ActiveFunc == "Relu":
            return numpy.maximum(0, X)
        else:
            raise ValueError("Unsupported activation function")
        
    def fit(self, X, Y):
        if self.RandomState is not None:
            numpy.random.seed(self.RandomState)

        SamplesNum, FeaturesNum = X.shape   # Shape of Training Dataset

        # Hidden Layer Weight (Random): Features Numer * Hidden Layer Neuron Number
        self.W = numpy.random.randn(FeaturesNum, self.HiddenLayerNeuronNum)
        # Hidden Layer Bias (Random):
        self.b = numpy.random.randn(self.HiddenLayerNeuronNum)
        # hidden layer output matrix
        #H = self._ActiveFunc(numpy.dot(X, self.W) + self.b)
        H = self._ActiveFunc(X @ self.W + self.b)
        # Moore–Penrose pseudo inverse
        H_pinv = numpy.linalg.pinv(H)
        # output weight
        #self.beta = numpy.dot(H_pinv, Y)
        self.beta = H_pinv @ Y

    def predict(self, X):
        H = self._ActiveFunc(X @ self.W + self.b)
        return H @ self.beta