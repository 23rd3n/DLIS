# **DLIS (Deep Learning on Inverse Problems)**

## **HW3:Optimization with Regularization**

- Observations are obtained through the convolution of a true vector x with a matrix A which is created from the 1D Gaussion Kernel. These observations are corrupted by n-dimensional iid Gaussian random noise vector. 

- In the coded solution, an analytical approach is used to calculate the MSE (Mean Squared-Error). This analytical approach uses SVD decomposition of the matrix A and gets an estimate of the true signal x by minimizing the reqularized least-squares estimator with the regularizer of norm-2 magnitude. Then using this expected value over random noise of norm-2 magnitude of difference between the estimate and true signal, a closed form solution is obtained. 

- Then from this closed form solution bias and variance terms are seperated to observe the trade of between. 

## **HW4:Denoising Images with a U-Net**

- In this homework, an end-to-end Unet neural network was trained on the Berkeley Segmentation Dataset (BSD300) which contains 300 clean color images. The dataset was seperated into 200 images for training dataset, 50 for testing dataset and the remaining 50 for the validation dataset.

- Firstly, for training purposes, all color images were converted into grayscale, then a zero-mean Gaussian noise was added to these grayscale images. Then these images were divided into chunks to reduce the computational costs.

- For validation and testing, we did not split images into chunks since it wasn't needed to compute the gradients.

- Then using DataLoader from the torch library, the dataset was prepared for training using a batch size for the epochs.

- After, the parameters of the Unet, loss function, and optimizer were selected. The model was trained on the normalized images checking the PSNR value and loss on validation to observe overfitting.

## **HW5: CT Image Reconstruction with a Variational Network**

### Unrolled Neural Networks:

- Improve upon a classic iterative method in terms of computational speed, or obtaining better image quailty.

- Considered linear inverse problem **y = Ax+b** , where A is typically wide matrix, or a matrix that is poorly conditioned, and z is the noise.

- Iterative Shrinkage Thresholding Algorithm (ISTA) is a fast iterative for solving the l1-reqularized least-squares optimization problem  
$$ \underset{x}{\min} \|kAx - y\|_2^2 + \lambda\|x\|_1 $$
for sparse vector reconstruction. ISTA is initilazed with $x_{0}=0$ and its iterations are given by

$$x_{t+1}=\tau_{\lambda \eta}x_{t}-\eta A^{T}(Ax_{t}-y)$$

where $\tau_{\eta}$ is the soft-thresholding operator (non-linearity).

- The motivation of the unrolled networks is to accelerate the existing ISTA algorithm through training but not necessarily better performance.

- Formulation of ISTA as a neural network: 

$$x_{t+1}=\tau_{\theta}(Qx_{t}+By)$$

with $Q=(I-\etaA^{T}A)$, $B=\etaA^{T}$, and $\theta=\lambda \eta$. This corresponds to performing a forward-pass through a recurrent neural network (unlike traditional NNs, RNNs have a memory component that enables them to maintain information about previous inputs and use it to make predictions or decisions), with fixed weights.

- Gregor and LeCun’s idea is to view the final result after k iterations, $f_{\theta}(y) = x_{t}$ , as function of the parameters $\theta = {Q, B, \theta}$, and train those parameters on a given dataset by minimizing the loss. This algorithm is called **LISTA** for **Learned ISTA**.

- Gregor and LeCun used a T=7 iterations and get the similar performance of ISTA with 70 iterations (10 times speedup). But if we don't care about the speed and time then running ISTA for so long will achieve a better performance than LISTA for sparse reconstruction.

- This algorithm has a caveat as can be seen from the equation $x_{t+1}=\tau_{\theta}(Qx_{t}+By)$, this algorithm doesn't take the forward model into account (we learn Q,B,$\theta$). Therefore algorithm both need to learn signal and forward model which is very wasteful. Also, for large size of images, Q and B is very large since occupying so much memory. Eventhough the memory requirement can be decreased using SVD or similar decompositions, not using the knowledge of forward model is not a favorable thing to do.