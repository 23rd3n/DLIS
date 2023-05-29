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

### 1- Unrolling the ISTA :

- Improve upon a classic iterative method in terms of computational speed, or obtaining better image quailty.

- Considered linear inverse problem **y = Ax+b** , where A is typically wide matrix, or a matrix that is poorly conditioned, and z is the noise.

- Iterative Shrinkage Thresholding Algorithm (ISTA) is a fast iterative for solving the l1-reqularized least-squares optimization problem 

    $$\min_{x} \frac{1}{2} \lVert Ax-y \rVert _{2}^{2}+ \lambda\lVert x \rVert _{1}$$

    for sparse vector reconstruction. ISTA is initilazed with $x_{0}=0$ and its iterations are given by

    $$x_{t+1}=\tau_{\lambda \eta}x_{t}-\eta A^{T}(Ax_{t}-y)$$

    where $\tau_{\eta}$ is the soft-thresholding operator (non-linearity).

- The motivation of the unrolled networks is to accelerate the existing ISTA algorithm through training but not necessarily better performance.

- Formulation of ISTA as a neural network: 

    $$x_{t+1}=\tau_{\theta}(Qx_{t}+By)$$

    with $Q=(I-\eta A^{T}A)$, $B=\eta A^{T}$, and $\theta=\lambda \eta$. This corresponds to performing a forward-pass through a recurrent neural network (unlike traditional NNs, RNNs have a memory component that enables them to maintain information about previous inputs and use it to make predictions or decisions), with fixed weights.

- Gregor and LeCun’s idea is to view the final result after k iterations, $f_{\theta}(y) = x_{t}$ , as function of the parameters $\theta = {Q, B, \theta}$, and train those parameters on a given dataset by minimizing the loss. This algorithm is called **LISTA** for **Learned ISTA**.

- Gregor and LeCun used a T=7 iterations and get the similar performance of ISTA with 70 iterations (10 times speedup). But if we don't care about the speed and time then running ISTA for so long will achieve a better performance than LISTA for sparse reconstruction.

- This algorithm has a caveat as can be seen from the equation $x_{t+1}=\tau_{\theta}(Qx_{t}+By)$, this algorithm doesn't take the forward model into account (we learn Q,B, $\theta$ ). Therefore algorithm both need to learn signal and forward model which is very wasteful. Also, for large size of images, Q and B is very large since occupying so much memory. Eventhough the memory requirement can be decreased using SVD or similar decompositions, not using the knowledge of forward model is not a favorable thing to do.

- Gregor and LeCun's work applies for sparse recovery but the idea of unrolling a proximal gradient descent algorithm can be applied to general iterative algorithms for solving a variety of problems.

### 2- A parameter efficent variational network :

- Unrolling the total-variation-reqularized least-squares, works well for MRI, has few params.

- To prevent over-fitting to the noisy data, it is beneficial to stop the ISTA after some iterations, but instead of early stopping we can also extend the LS problem by an additional regularization term R(u) to prevent over-fitting. There is a trade-off between regularizator and LS term.

- A popular choice for the regularizer for imaging problems is the total-variation norm, which works well if images consist of patches that are relatively constant. The total variation norm:

    $$\lVert x \rVert  = \sum_{i=1}^{n-1} |x_{i}-x_{i+1}|$$

    This can be written as, $\lVert x \rVert _{TV} = \lVert Cx \rVert _{1}$ where C is the circular matrix in the first row c=[1,-1,0,...,0]. Total Variation norm can be viewed as convolution with the filter [-1,1] then summing up the entries:

    $$\lVert x \rVert _{TV} = \langle |Cx|,1 \rangle$$
    where 1 = [1,1,1..,1] is all-ones vector.

- A generalization of the TV-nrom is the Fields of Expert Model defined as:

$$R(x) = \sum_{i=1}^{k} \langle \phi_{i} (C_{i} x), 1 \rangle$$

- The Fields of Expert models can be better regularizer for natural images than the total variation norm, since it contains TV as a special case, but can also express more complex regularizers.

- **The variational network is obtained by unrolling the gradient descent iterations of a regularized least-squares problem with a Fields of Expert regularizer. Then the overall equation would be**

    $$x_{t+1} = x_{t} - \eta (A^{T} (A x_{t} - y) + \sum_{i=1}^{k} C^{T} \psi' (C_{i} x_{t}))$$

    The last part can be viewed as a two-layer CNN, where the trainable parameters are the convolutional filters and the parameters of the activation function. Note that the parameters in each of the unrolled iteartions are trainable, and the stepsize parameter $\eta _{t}$ is also trainable as well.

- The variational network discussed here outperforms a traditional un-trained method like TV norm minimization on natural and medical images and is relatively paramater efficient. However, substituting the simple two-layer CNN with a large U-net significantly improves the performance and gives essentially state-of-the-art performance for MRI reconstruction.


## **HW6: Noise2Noise vs. Supervised Learning

- Supervised training requries $(x_{i},y_{i})$ data. A NN $f_{\theta}$ is trained to predict target image based on noisy measuremnt by minimizing the supervised loss:

$$ L(\theta) = \frac{1}{N} \sum_{i=1}^{N} l(f_{\theta} (y_{i}, x_{i}))$$

    where l is a loss function such as MSE or l1-norm.

- But in real-life practice, we don't always have access to the ground-truth, therefore we need different approaches to such problems.

- Three approaches for constructing self-supervised loss:

    - Construct an unbiased estimate of supervised loss

        - requires extra measurements but doesn't require extra assumptions and with sufficent data, gets the performance as in the supervised loss

    - Construct a self-supervised loss based on Stein's unbiased estimator.

    - Construct a self-supervised loss based on predicting one part of an image from another part, and as such makes explicit or implicit assumptions about images. Even with a lot of training data, this approach is typically performs worse than the supervised manner.

