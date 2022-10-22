import pandas as pd
import numpy as np
import time

import rust_bayes_module as bayes

# Basic Python interface for rust module
class classifier:

    def fit(self, X:pd.DataFrame, y:pd.DataFrame, p:pd.DataFrame = None, c:pd.DataFrame = None):
        """Determines the means, covariances and priors for the provided data set

        Args:
            `X (pd.DataFrame)`: Data set, samples as rows
            `y (pd.DataFrame)`: Labels for data set
            `p (pd.DataFrame, optional)`: Priors Defaults to None.
            `c (pd.DataFrame, optional)`: Subset of classes to use. Defaults to None.
        """
        
        # Check dimensions        
        if len(X) != len(y) :
            raise ValueError(f"Length of X and y do not match : len(X):{len(X)} != len(y):{len(y)}")
        
        # Get sorted list of all classses in data set        
        classes = np.sort(y.unstack().unique()) if c == None else c
        
        # Save dimensionality
        self.dim = X.shape[1]
    
        # Calculate mean, covariance and priors of data set
        self.M = np.ascontiguousarray([X[y[0] == c].mean() for c in classes], dtype = float)
        self.S = np.ascontiguousarray([X[y[0] == c].cov() for c in classes], dtype = float)
        self.P = np.ascontiguousarray([len(y[y[0] == c]) for c in classes], dtype = float) if p == None else p
        
    def predict(self,X:pd.DataFrame, verbose = True):
        """Predict the class of the given sample(s)

        Args:
            `X (pd.DataFrame)`: Data set to classify
            `verbose (bool)`: Print message with classification time. Defaults to True

        Returns:
            `(np.array)`: _description_
        """
        
        # Check if dimensions of new data set mathces model        
        if (d1:=self.dim) != (d2:=X.shape[1]) :
            raise ValueError(f"Incorrect number of features for this model. Expected {d1}, got {d2}")
        
        # Call rust module to make predictions
        return bayes.classifier( np.ascontiguousarray(X), self.M, self.S, self.P , verbose )