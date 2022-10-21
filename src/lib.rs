// #![allow(unused)]
#![allow(non_snake_case)]

use nalgebra::{self, Dynamic,Matrix, VecStorage, SliceStorage, Const,DMatrixSlice};
use numpy::{IntoPyArray,PyReadonlyArrayDyn,PyArray1,ndarray::ArrayViewD};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rayon::{slice::ParallelSlice, prelude::ParallelIterator};
use std::{f64::consts::PI, ops::Index};

#[pymodule]
#[pyo3(name = "rust_bayes_module")]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // Python wrapper
    #[pyfn(m)]
    #[pyo3(name = "classifier_multi")]
    fn bayesian_classifier_multi_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        m: PyReadonlyArrayDyn<f64>,
        s: PyReadonlyArrayDyn<f64>,
        p: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray1 <usize> {

        let av_pred = bayesian_classifier_multi(x.as_array(), m.as_array(), s.as_array(),p.as_array());

        let nd_pred = ndarray::Array1::from_shape_vec(ndarray::Dim(x.shape()[0]), av_pred).unwrap();

        nd_pred.into_pyarray(py)
    }

    Ok(())
}

struct ClassData <'a>  {
    mean : Matrix<f64, Dynamic, Dynamic, SliceStorage<'a, f64, Dynamic, Dynamic, Const<1>, Dynamic>>,
    sinv : Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    norm : f64,
    prio : f64,
}

/// Multi-threaded naive bayesian classifier
fn bayesian_classifier_multi (
    X : ArrayViewD<'_, f64>,
    M : ArrayViewD<'_, f64>,
    S : ArrayViewD<'_, f64>,
    P : ArrayViewD<'_, f64>,
)  ->  Vec<usize> {

    // Check for number of class
    if S.shape()[0] != M.shape()[0] || P.shape()[0] != M.shape()[0] {
        panic!("Number of mean vectors and covariance matrices are not equal. Got {} means and {} covariances.",M.shape()[0],S.shape()[0])
    }

    // Check for number square covariances
    if S.shape()[1] != S.shape()[2] {
        panic!("Covariance matrices need to be square. Got {} x {}",S.shape()[1], S.shape()[2])
    }

    // Check for number dimensions being equal
    if X.shape()[1] != M.shape()[1] || S.shape()[1] != M.shape()[1] {
        panic!("Inputs have varying number of dimensions. Got Dim(X) = {:?}, Dim(M) = {:?}, Dim(S) = {:?}",X.shape()[1],M.shape()[1],S.shape()[1])
    }

    // Save important shape parameters
    let C = M.shape()[0]; // Number of unique classes
    let D = M.shape()[1]; // Number of dimensions

    // Ensure slices are valid
    let (X_slice,M_slice,S_slice,P_slice) = match (X.to_slice(),M.to_slice(),S.to_slice(),P.to_slice()) {
        (Some(x), Some(m), Some(s), Some(p)) => (x,m,s,p),
        _ => panic!("One or more array slices are incompatible, need to be contiguous and in standard order"),
    };

    // Reshape inputs into correctly sized nalgebra matrices
    let m = DMatrixSlice::from_slice(M_slice ,D,C);
    let s = DMatrixSlice::from_slice(S_slice ,D,D*C).transpose();

    // prepare vector containing data for each classs (mean,inv_cov,norm)
    let mut class : Vec < ClassData > = Vec::with_capacity(C);

    // Do initial preparation for each class
    for c in 0..C {

        // Form matrix from slice
        let mean = m.slice((0,c), (D,1));
        let scov = s.slice((c*D,0), (D,D));

        let prio = *P_slice.index(c);
        
        // Get inverse of s
        let sinv = match scov.try_inverse() {
            Some(inv) => inv,
            None => {panic!("Failed to invert matrix for class {c} : {}",scov)},
        };

        // Calculate normalization factor
        let norm = 1.0/((2.0*PI).powf(D as f64/2.0)*scov.determinant().sqrt());

        // Add class data to vector
        class.insert(c,ClassData{ mean, sinv, norm, prio } );
    }

    // Get vector of predictions
    let predictions = X_slice.par_chunks(D)
    .map_init( || vec![0.0;C], |x_probabilities,sample| {
        
        let x_sample = DMatrixSlice::from_slice(sample,D,1);

        // Calculate multivariate normal pdf and save probability for each class
        for (c_i,c) in class.iter().enumerate() {
            x_probabilities[c_i] = c.norm*(-0.5*(x_sample-c.mean).transpose()*&c.sinv*(x_sample-c.mean)).exp()[0]*c.prio;
        }

        // Run argmax on the probability array
        match x_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id) {
                Some(k) => k,
                None => panic!("Failed during argmax operation."),
            }
    } ).collect();

    // Return predictions
    predictions

}
