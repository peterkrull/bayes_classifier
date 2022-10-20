import pandas as pd
import numpy as np
import time
import sys

# sys.path.append("/home/peterkrull/multivariate_gauss_pdf/target/release")
import naive_bayes_module as bayesian

def classifier(x:pd.DataFrame,m:list[np.array],s:list[np.array],target:str = None,multi=True):
    """Naive Bayesian classification of entire data sets. Built in rust for fearless concurrency.

    Args:
        `x (pd.DataFrame)`: Pandas dataframe containing the samples (row-wise) to classify.
        `m (list[np.array])`: Python list with class means as numpy arrays
        `s (list[np.array])`: Python list with class covariances as numpy arrays
        `target (str, optional)`: Key of column containing class labels for all samples. Defaults to None.
        `multi (bool, optional)`: Wether to use multi or single threaded rust function. Defaults to True.

    Returns:
        `np.array`: array of estimated classes
        or, if target is supplied
        `np.array` , `np.array`: array of estimated classes and confusion matrix
    """
    start = time.time()
    if target:
        X = np.ascontiguousarray(x.drop(target,axis=1).to_numpy())
    else:
        X = np.ascontiguousarray(x.to_numpy())
        
    M = np.ascontiguousarray(m)
    S = np.ascontiguousarray(s)
           
    if multi:
        est = bayesian.classifier_par(X,M,S)
    else:
        est = bayesian.classifier_single(X,M,S)
        
    if target:
        tru = x.get('target')
        conf_matrix2 = np.zeros((len(tru.unique()),len(tru.unique())))
        for (t,e) in zip(tru,est):
            conf_matrix2[int(t),int(e)] += 1
        print(f"Classification took : {round(time.time()-start,3)} seconds")
        return est,conf_matrix2
    else:
        print(f"Classification took : {round(time.time()-start,3)} seconds")
        return est