import numpy as np

# I used this githubs to get a concept of how to write the code
#https://github.com/Natsu6767/Adaline
#https://github.com/camilo-v/Neural-Networks


class AdalineAlgo:
    def __init__(self, rate=0.01, niter=15, shuffle=True):
        self.learning_rate = rate
        self.niter = niter
        self.shuffle = shuffle

        # vector that represents the Weights
        self.weight = []

        # Cost function
        self.cost_ = []


    #We use this function to train our model, and we update the weights for each iteration.
    def fit(self, X, y):
        row = X.shape[0]
        col = X.shape[1]

        #  add bias to X
        X_bias = np.ones((row, col + 1))
        X_bias[:, 1:] = X
        X = X_bias

        # initialize the weights
        np.random.seed(1)
        self.weight = np.random.rand(col + 1)

        # training
        for _ in range(self.niter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def _update_weights(self, xi, target):
   
        output = self.net_input(xi)
        error = target - output
        """
        Note:the bias calc (= self.weight[0]) will be equal to: 
        self.weight[0] += self.learning_rate * errors
        because We defined: X.T[0] = 1
        """
        self.weight += self.learning_rate * xi.dot(error)
        cost = 0.5 * (error ** 2)
        return cost

    #to adjust the training data with np random permutation
    def _shuffle(self, X, y):
        per = np.random.permutation(len(y))
        return X[per], y[per]

    #here we used Matrix multiplication to calculate the net input.
    def net_input(self, X):
        return X @ self.weight
    
    #Linear activation for a static function.
    def activation(self, X):
        return self.net_input(X)
    

    #Return class label after unit step.
    def predict(self, X):
      
        # if x is list instead of np.array
        if type(X) is list:
            X = np.array(X)

        # add bias to x if he doesn't exist
        if len(X.T) != len(self.weight):
            X_bias = np.ones((X.shape[0], X.shape[1] + 1))
            X_bias[:, 1:] = X
            X = X_bias

        return np.where(self.activation(X) > 0.0, 1, -1)

    #gives us the success rate of our model : its calculated based on comparison of
        expected value and predicted value.
    def score(self, X, y):
        wrong_prediction = abs((self.predict(X) - y) / 2).sum()
        self.score_ = (len(X) - wrong_prediction) / len(X)
        return self.score_
