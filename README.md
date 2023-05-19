# **DLIS (Deep Learning on Inverse Problems)**

## **HW3**

- Observations are obtained through the convolution of a true vector x with a matrix A which is created from the 1D Gaussion Kernel. These observations are corrupted by n-dimensional iid Gaussian random noise vector. 

- In the coded solution, an analytical approach is used to calculate the MSE (Mean Squared-Error). This analytical approach uses SVD decomposition of the matrix A and gets an estimate of the true signal x by minimizing the reqularized least-squares estimator with the regularizer of norm-2 magnitude. Then using this expected value over random noise of norm-2 magnitude of difference between the estimate and true signal, a closed form solution is obtained. 

- Then from this closed form solution bias and variance terms are seperated to observe the trade of between. 

## **HW4**

- In this homework, an end-to-end Unet neural network was trained on the Berkeley Segmentation Dataset (BSD300) which contains 300 clean color images. The dataset was seperated into 200 images for training dataset, 50 for testing dataset and the remaining 50 for the validation dataset.

- Firstly, for training purposes, all color images were converted into grayscale, then a zero-mean Gaussian noise was added to these grayscale images. Then these images were divided into chunks to reduce the computational costs.

- For validation and testing, we did not split images into chunks since it wasn't needed to compute the gradients.

- Then using DataLoader from the torch library, the dataset was prepared for training using a batch size for the epochs.

- After, the parameters of the Unet, loss function, and optimizer were selected. The model was trained on the normalized images checking the PSNR value and loss on validation to observe overfitting.
