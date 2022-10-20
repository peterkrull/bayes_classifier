// #![allow(unused)]
#![allow(non_snake_case)]

use nalgebra::{self, Dynamic,Matrix, VecStorage, SliceStorage, Const};
use numpy::{IntoPyArray,PyReadonlyArrayDyn,PyArray1,ndarray::ArrayViewD};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rayon::{slice::ParallelSlice, prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator}};
use std::f64::consts::PI;

#[pymodule]
#[pyo3(name = "naive_bayes")]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // Python wrapper
    #[pyfn(m)]
    #[pyo3(name = "classifier_multi")]
    fn bayesian_classifier_multi_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        m: PyReadonlyArrayDyn<f64>,
        s: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray1 <usize> {

        let av_pred = bayesian_classifier_multi(x.as_array(), m.as_array(), s.as_array());

        let nd_pred = ndarray::Array1::from_shape_vec(ndarray::Dim(x.shape()[0]), av_pred).unwrap();

        nd_pred.into_pyarray(py)
    }

    // Python wrapper
    #[pyfn(m)]
    #[pyo3(name = "classifier_single")]
    fn bayesian_classifier_single_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        m: PyReadonlyArrayDyn<f64>,
        s: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray1 <usize> {

        let av_pred = bayesian_classifier_single(x.as_array(), m.as_array(), s.as_array());

        let nd_pred = ndarray::Array1::from_shape_vec(ndarray::Dim(x.shape()[0]), av_pred).unwrap();

        nd_pred.into_pyarray(py)
    }

    Ok(())
}

/// Multi-threaded naive bayesian classifier
fn bayesian_classifier_multi (
    X : ArrayViewD<'_, f64>,
    M : ArrayViewD<'_, f64>,
    S : ArrayViewD<'_, f64>,
)  ->  Vec<usize> {

    // Check for number of covariances
    if S.shape()[0] != M.shape()[0] {
        panic!("Number of mean vectors and covariance matrices are not equal. Got {} means and {} covariances.",M.shape()[0],S.shape()[0])
    }

    // Check for number square covariances
    if S.shape()[1] != S.shape()[2] {
        panic!("Covariance matrices need to be square. Got {} x {}",S.shape()[1], S.shape()[2])
    }

    // Check for number square covariances
    if X.shape()[1] != M.shape()[1] || S.shape()[1] != M.shape()[1] {
        panic!("Inputs have varying number of dimensions. Got Dim(X) = {:?}, Dim(M) = {:?}, Dim(S) = {:?}",X.shape()[1],M.shape()[1],S.shape()[1])
    }

    // Save important shape parameters
    let C = M.shape()[0]; // Number of unique classes
    let N = X.shape()[0]; // Number of samples in X
    let D = M.shape()[1]; // Number of dimensions

    // Ensure slices are valid
    let (X_slice,M_slice,S_slice) = match (X.to_slice(),M.to_slice(),S.to_slice()) {
        (Some(x), Some(m), Some(s)) => (x,m,s),
        _ => panic!("One or more array slices are incompatible, need to be contiguous and in standard order"),
    };

    // Reshape inputs into correctly sized nalgebra matrices
    let m = nalgebra::DMatrixSlice::from_slice(M_slice ,D,C);
    let s = nalgebra::DMatrixSlice::from_slice(S_slice ,D,D*C).transpose();

    // prepare vector containing data for each classs (mean,inv_cov,norm)
    let mut class : Vec< (
        Matrix<f64, Dynamic, Dynamic, SliceStorage<f64, Dynamic, Dynamic, Const<1>, Dynamic>>,
        Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
        f64
    ) > = Vec::with_capacity(C);

    // Do initial preparation for each class
    for c in 0..C {

        // Form matrix from slice
        let m_c = m.slice((0,c), (D,1));
        let s_c = s.slice((c*D,0), (D,D));
        
        // Get inverse of s
        let s_in = match s_c.try_inverse() {
            Some(inv) => inv,
            None => {panic!("Failed to invert matrix for class {c} : {}",s_c)},
        };

        // Calculate normalization factor
        let norm = 1.0/((2.0*PI).powf(D as f64/2.0)*s_c.determinant().sqrt());

        // Add class data to vector
        class.insert(c,(m_c,s_in,norm));
    }

    // Vectors for predictions and probabilities
    let mut predictions :Vec<usize> = vec![0;N];
    // let mut probabilities :Vec<f64> = vec![0.0;C];

    X_slice.par_chunks(D).into_par_iter().zip(&mut predictions).for_each(|(sample,pred)| {
        // TODO : This allocation might not be very effecient to do every iteration. Some sort of pre allocation would be better
        let mut x_probabilities : Vec<f64> = vec![0.0;C];

        let x_sample = nalgebra::DMatrixSlice::from_slice(sample,D,1);

        for (c_i,c) in class.iter().enumerate() {

            // Calculate MVN PDF and save probability in vector
            x_probabilities[c_i] = c.2*(-0.5*(x_sample-&c.0).transpose()*&c.1*(x_sample-&c.0)).exp()[0];
        }

        // Run argmax on the probability array
        *pred = match x_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id) {
                Some(k) => k,
                None => panic!("Failed during argmax operation."),
            } ; 
    } );

    // Return predictions
    predictions

}


/// Single threaded naive bayesian classifier
fn bayesian_classifier_single (
    X : ArrayViewD<'_, f64>,
    M : ArrayViewD<'_, f64>,
    S : ArrayViewD<'_, f64>,
)  ->  Vec<usize> {

    // Check for number of covariances
    if S.shape()[0] != M.shape()[0] {
        panic!("Number of mean vectors and covariance matrices are not equal. Got {} means and {} covariances.",M.shape()[0],S.shape()[0])
    }

    // Check for number square covariances
    if S.shape()[1] != S.shape()[2] {
        panic!("Covariance matrices need to be square. Got {} x {}",S.shape()[1], S.shape()[2])
    }

    // Check for number square covariances
    if X.shape()[1] != M.shape()[1] || S.shape()[1] != M.shape()[1] {
        panic!("Inputs have varying number of dimensions. Got Dim(X) = {:?}, Dim(M) = {:?}, Dim(S) = {:?}",X.shape()[1],M.shape()[1],S.shape()[1])
    }

    // Save important shape parameters
    let C = M.shape()[0]; // Number of unique classes
    let N = X.shape()[0]; // Number of samples in X
    let D = M.shape()[1]; // Number of dimensions

    // Ensure slices are valid
    let (X_slice,M_slice,S_slice) = match (X.to_slice(),M.to_slice(),S.to_slice()) {
        (Some(x), Some(m), Some(s)) => (x,m,s),
        _ => panic!("One or more array slices are incompatible, need to be contiguous and in standard order"),
    };
  
    // Reshape inputs into correctly sized nalgebra matrices
    let x = nalgebra::DMatrixSlice::from_slice(X_slice ,D,N);
    let m = nalgebra::DMatrixSlice::from_slice(M_slice ,D,C);
    let s = nalgebra::DMatrixSlice::from_slice(S_slice ,D,D*C).transpose();

    // prepare vector containing data for each classs (mean,inv_cov,norm)
    let mut class : Vec< (
        Matrix<f64, Dynamic, Dynamic, SliceStorage<f64, Dynamic, Dynamic, Const<1>, Dynamic>>,
        Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
        f64
    ) > = Vec::with_capacity(C);

    // Do initial preparation for each class
    for c in 0..C {

        // Form matrix from slice
        let m_c = m.slice((0,c), (D,1));
        let s_c = s.slice((c*D,0), (D,D));
        
        // Get inverse of s
        let s_in = match s_c.try_inverse() {
            Some(inv) => inv,
            None => {panic!("Failed to invert matrix for class {c} : {}",s_c)},
        };

        // Calculate normalization factor
        let norm = 1.0/((2.0*PI).powf(D as f64/2.0)*s_c.determinant().sqrt());

        // Add class data to vector
        class.insert(c,(m_c,s_in,norm));
    }

    // Vectors for predictions and probabilities
    let mut predictions :Vec<usize> = Vec::with_capacity(N);
    let mut probabilities :Vec<f64> = vec![0.0;C];

    // Iterate through each sample for each class and evaluate the MVN PDF
    for (x_i,x_sample) in x.column_iter().enumerate() {
        for (c_i,c) in class.iter().enumerate() {

            // Calculate MVN PDF and save probability in vector
            probabilities[c_i] = c.2*(-0.5*(x_sample-&c.0).transpose()*&c.1*(x_sample-&c.0)).exp()[0];
        }

        // Run argmax on the probability array
        predictions.insert(x_i , match probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index) {
            Some(pred) => pred,
            None => panic!("Failed during argmax operation."),
        } );

    }

    // Return predictions
    predictions

}
