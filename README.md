# Naive Bayes Classifer built in Rust

Simple Python-module built in Rust to classify large data sets concurrently in seconds using the Naive Bayes Classifier. Building the module in rust allows for effecient and concurrent evaluation of the samples. Orders of magnitude faster than doing the same in pure Python.

## Usage

The most basic use of the classifier involes a set of samples and the means and covariances of the classes. 

``` python
import naive_bayes

X = # DataFrame containing data to classify 
M = # List of class means as numpy.array
S = # List of class covariances as numpy.array

est = naive_bayes.classifier(X,M,S)
```
An alternative use involved providing a string to the classifier, signifying the name of the column containing the labels of each sample. This causes the function to also return a confusion matrix, such that the performance of the classification can be evaluated easily.

``` python
est, conf = naive_bayes.classifier(X,M,S,'target')
```
