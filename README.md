# DLIS
## The solutions created by me to keep track 
### Starts from the HW3

** HW3 **
- Observations are obtained through the convolution of a true vector x with a matrix A which is created from the 1D Gaussion Kernel. These observations are corrupted by n-dimensional iid Gaussian random noise vector. 
- In the coded solution, an analytical approach is used to calculate the MSE (Mean Squared-Error). This analytical approach uses SVD decomposition of the matrix A and gets an estimate of the true signal x by minimizing the reqularized least-squares estimator with the regularizer of norm-2 magnitude. Then using this expected value over random noise of norm-2 magnitude of difference between the estimate and true signal, a closed form solution is obtained. 
- Then from this closed form solution bias and variance terms are seperated to observe the trade of between. 

