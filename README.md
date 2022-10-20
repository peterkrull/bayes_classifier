# Naive Bayes Classifier built in Rust

Simple Python-module built in Rust to classify large data sets concurrently in seconds using Naive Bayes Classification. Building the module in rust allows for effecient and concurrent evaluation of the samples, making this implementation orders of magnitude faster than doing the same in pure Python. Speaking from experience on that one.

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

----

## To-Do

- Currently priors are not taken into account, meaning that a given sample is equally likely to be any class. This may not be the case in reality, and thus priors need to be implemented. This could be implemented simply by modifying the function arguments, such theat the last example above becomes:

```python
est = naive_bayes.classifier(X,M,S,P)
```

- Improve memory allocation during parallelization. Some allocations happen in a loop, which may hinder performacne slightly. Preallocating should improve performance slightly.
