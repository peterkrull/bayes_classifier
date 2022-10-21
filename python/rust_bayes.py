import pandas as pd
import numpy as np
import time
import sys

# sys.path.append("/home/peterkrull/multivariate_gauss_pdf/target/release")
import rust_bayes_module as bayes

def classifier(x:pd.DataFrame,m:list[np.array],s:list[np.array],p:list[float] = None,target:str = None,uniform_priors=False):
    """Bayesian classification of entire data sets. Built in rust for fearless concurrency.

    Args:
        `x (pd.DataFrame)`: Pandas dataframe containing the samples (row-wise) to classify.
        `m (list[np.array])`: Python list with class means as numpy arrays
        `s (list[np.array])`: Python list with class covariances as numpy arrays
        `p (list[float], optional)`: Python list with priors, can be determined form data if `target` is supplied
        `target (str, optional)`: Key of column containing class labels for all samples. Defaults to None.

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
    P = np.ascontiguousarray(p)
    
    # Determine priors
    if type(p).__module__ == np.__name__:
        P = np.ascontiguousarray(p)
    elif target and not uniform_priors:
        classes = np.sort(x['target'].unique())
        P = np.ascontiguousarray([len(x[x[target] == c]) for c in classes])
        print(f"Warning : No priors were supplied, assuming priors from target column :\n{P/len(x['target'])}")
    else:
        if uniform_priors:
            print("Using uniform priors")
        else:
            print("Warning : No priors were supplied, assuming uniform priors.")
        P = np.ascontiguousarray([1]*S.shape[0])
           
    est = bayes.classifier_multi(X,M,S,P)
        
    if target:
        
        # Calculate confusion matrix
        tru = x.get('target')
        conf_matrix = np.zeros((len(tru.unique()),len(tru.unique())), dtype=int)
        np.add.at(conf_matrix, (tru.to_numpy(dtype = int), est), 1)
        
        print(f"Classification took : {round(time.time()-start,3)} seconds")
        return est,conf_matrix
    else:
        print(f"Classification took : {round(time.time()-start,3)} seconds")
        return est