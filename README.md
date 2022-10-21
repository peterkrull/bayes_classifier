# Bayes Classifier built in Rust

Simple Python-module built in Rust to classify large data sets concurrently in seconds using Bayes Classification. Compared with Naive Bayesian classification, where it is assumed that the covariance matrix only has diagonal entries, a "full" Bayesian classification uses the entire covariance matrix, allowing for correlated samples. Building the module in rust allows for effecient and concurrent evaluation of the samples, making this implementation orders of magnitude faster than doing the same in pure Python. Speaking from experience on that one.

## Usage

The most basic use of the classifier involved fitting a gaussian distribution to each class based on a labeled data set.

``` python
from rust_bayes import classifier_class

X = # DataFrame containing data to classify 
y = # DataFrame containing data labels 

# Get classifier object
bayes = rust_bayes.classifier()

# Fit the model to a data set
bayes.fit(X,y)

# Predict 
est = bayes.predict(X_test)
```

Alternatively, given a list of means and a list of covariances, a set of samples can be evaluated as such:

```python
import rust_bayes

X = # DataFrame containing data to classify 
M = # List of class means as numpy.array
S = # List of class covariances as numpy.array
P = # Class priors as a numpy array

est = rust_bayes.classifier(X,M,S,P)

# If priors are not available, uniform priors are assumed:
est = rust_bayes.classifier(X,M,S)
```
It is also possible to provide a string to the classification function, signifying the name of the column containing the labels of each sample in `X`. This causes the function to also return a confusion matrix, such that the performance of the classification can be evaluated easily.

``` python
est, conf = rust_bayes.classifier(X,M,S,P,'target')

# Omitting the priors argument results in the priors being determined from target column of X
est, conf = rust_bayes.classifier(X,M,S,target='target')
```
