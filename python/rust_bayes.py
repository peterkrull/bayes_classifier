import numpy as np

import rust_bayes_module as bayes

# Basic Python interface for rust module
class classifier:

    def fit(self, X:'np.ndarray', y:'np.ndarray', p:'np.ndarray' = None, c:'np.ndarray' = None):
        """Determines the means, covariances and priors for the provided data set

        Args:
            `X (np.ndarray)`: Data set, samples as rows
            `y (np.ndarray)`: Labels for data set
            `p (np.ndarray, optional)`: Priors Defaults to None.
            `c (np.ndarray, optional)`: Subset of classes to use. Defaults to None.
        """
        
        # If DataFrame is proveded, convert into numpy array
        if type(X).__name__ == 'DataFrame' : X = np.array(X)
        if type(y).__name__ == 'DataFrame' : y = np.array(y).flatten()
        if type(p).__name__ == 'DataFrame' : p = np.array(p).flatten()
        if type(c).__name__ == 'DataFrame' : c = np.array(c).flatten()

        # Check dimensions        
        if (l1:=len(X)) != (l2:=len(y)) :
            raise ValueError(f"Length of X and y do not match : len(X):{l1} != len(y):{l2}")
        
        # Get sorted list of all classses in data set        
        self.classes = np.unique(y) if c == None else c
        
        # Save dimensionality
        self.dim = X.shape[1]
    
        # Calculate mean, covariance and priors of data set
        self.M = np.ascontiguousarray( [np.mean(X[y == c],axis=0) for c in self.classes], dtype = float )
        self.S = np.ascontiguousarray( [np.cov(X[y == c],rowvar=False) for c in self.classes], dtype = float )
        self.P = np.ascontiguousarray( [len(y[y == c]) for c in self.classes], dtype = float ) if p == None else p
        
        if (cond:=max([np.linalg.cond(s) for s in self.S])) > 1000:
            print(f"Warning: High conditioning number : {round(cond,2)}")
        
    def predict(self,X:'np.ndarray', verbose = True) -> 'np.ndarray':
        """Predict the class of the given sample(s)

        Args:
            `X (np.ndarray)`: Data set to classify
            `verbose (bool)`: Print message with classification time. Defaults to True

        Returns:
            `(np.ndarray)`: _description_
        """
        
        # If DataFrame is proveded, convert into numpy array
        if type(X).__name__ == 'DataFrame' : X = np.array(X)
        
        # Check if dimensions of new data set mathces model        
        if (d1:=self.dim) != (d2:=X.shape[1]) :
            raise ValueError(f"Incorrect number of features for this model. Expected {d1}, got {d2}")
        
        # Call rust module to make predictions
        return self.classes[bayes.classifier( np.ascontiguousarray(X), self.M, self.S, self.P , verbose )]
