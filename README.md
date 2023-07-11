# **DLIS (Deep Learning on Inverse Problems)**
**For detailed explanations of the codes, please refer the README inside the homework folders.**
## **HW1:Denoising a signal that lies in a subspace by projecting the observations onto this subspace**
- We have some observations y, and we have a orthanormal basis matrix for k-dimensional subspace of n-dimensional signal x. 
- For some values of k (from 0 to 1000 increasing by 100), we analyze the denoising performance using the average mean-squared error. 

## **HW2: Implementation of ISTA(Iterative Soft-Thresholding Algorithm)**

- Iterative Shrinkage Thresholding Algorithm (ISTA) is a fast iterative for solving the l1-reqularized least-squares optimization problem 

    $$\min_{x} \frac{1}{2} \lVert Ax-y \rVert _{2}^{2}+ \lambda\lVert x \rVert _{1}$$

    for sparse vector reconstruction. ISTA is initilazed with $x_{0}=0$ and its iterations are given by

    $$x_{t+1}=\tau_{\lambda \eta}x_{t}-\eta A^{T}(Ax_{t}-y)$$

    where $\tau_{\eta}$ is the soft-thresholding operator (non-linearity).

- By minimizing the cost function via ISTA, we'll reconstruct 100-sparse signal x and we'll investigate the convergence of the algorithm. Then we'll change the sparsity and analyze if we are still able to reconstruct the sparse signal.

- Signal models are very important for inverse problems because in inverse problems the forward map is not invertible or is ill-conditioned (ratio of singular values --> infinity). Therefore, either there are many solutions possible (more columns than rows) or since it is ill-conditioned it is not favorable to invert it (noise amplification).
- Signal models help us escape these problems. With the assumption on the signal to be reconstructed, we can select one of the solutions in where there are many possible.
- One of those signal models are the assumption of signal lies in the subspace or in the union of subspaces. Then using a few measurements or observations we can reconstruct the original signal. Therefore, sparsity in the signals are very important for computational efficacy and sample efficacy. Even if the signals are not sparse in some domains, we can find some dictionaries (domains) where they are sparse. For example, natural images are sparse in Discrete Cosine Transform. 
- In natural images, low-frequency coefficients posses the most energy therefore ordered DCT coefficients decay very quickly that even we use 10% of them we would be still able to see the image details clearly. JPEG and lossy compression use this idea for size reduction. This is an example of orthanormal dictionaries as the norm of the dictionary is always 1 for DCT.
- There are also overcomplete dictionaries. These dictionaries are such that even if you remove one or many columns, you would still be able to span the whole space. Therefore, they are called overcomplete.
- Overcomplete dictionaries are more efficient in a sense that it is easier to find an overcomplete dictionary than to find an orthanormal bases. However, when representing signal in a overcomplete dictionaries, then reconstruction of the signals back is difficult since there are many representations of the sparse signal in the original.
- In general we want to recover,  y = A'x'= A'Dx = Ax, here x is sparse in D and A=A'D, we can think of A as the subsampled Fourier Transform for MRI and D as the wavelet transform for sparse representation. Therefore, at the end we have y=Ax inverse problem that we want to find the x, which is a sparse vector.
- The best one would be to use the l0-norm minimization of x that are subject to y=Ax, but this problem is computationally intractable, NP-hard problem. Therefore, instead we use l1-norm minimization (also called Basis Pursuit) to minimize the cost function and get the sparse vector x. One such algorithm that is in this class is ISTA which we defined above. There is a short check to see if the Basis Pursuit will allow us to get a reconstruction which is if number of measurement is larger than 2*sparsity * log(#cols/sparsity) given that #cols are large and sparsity is mildy large. Otherwise we would need to check if the tangenct cone of l1-norm and null space of A only collide in 0-vector.
- The other algorithm is in the class of Greedy Methods, OMP (orthagonal matching pursuit), which builds a support set of sparse vector iteratively checking the correlations between residual vector and columns of A. This algorithm is easy to implement and computaitonally efficient but has a downside that if a wrong index is selected for the support set, then it stays in that set until the end. There are some advancements to this algorithm who take care of this problem.
- Also there is a good theorem for sparse reconstruction that if every 2s-many columns of A are linearly independent, then the sparse reconstruction is possible. 
- Finally, if a matrix A is mu-incoherent and s<(1/2*mu), then solution to both OMP and l1-norm minimization for y=Ax is the x itself. mu-coherence is defined as follows: if a matrix with unit norm columns has corelations with its every columns smaller than or equal to mu, then this matrix is called mu-incoherent.

## **HW3:Optimization with Regularization**

- Observations are obtained through the convolution of a true vector x with a matrix A which is created from the 1D Gaussion Kernel. These observations are corrupted by n-dimensional iid Gaussian random noise vector. 

- In the coded solution, an analytical approach is used to calculate the MSE (Mean Squared-Error). This analytical approach uses SVD decomposition of the matrix A and gets an estimate of the true signal x by minimizing the reqularized least-squares estimator with the regularizer of norm-2 magnitude. Then using this expected value over random noise of norm-2 magnitude of difference between the estimate and true signal, a closed form solution is obtained. 

- Then from this closed form solution bias and variance terms are seperated to observe the trade of between. 

- Optimization based reconstruction methods work well if the forward operator A and statistical model of the underlying noise e are known. Also for optimization based reconstruction to yield an unique solution the most important thing is PRIOR ASSUMPTION on the signal must be well-modelled mathematically. For example, for sparse reconstruction, l1-norm regularizer is a good choice to begin with.
- In the setups where these conditions are not satisfied, then Machine Learning methods may be well suited.
- For inverse problems with with uncorraleted homogenous gaussian noise, the MAP estimator for log-likelihood function yields the least squares minimization with the prior assumption log p(x) = lamda*R(x).
- The choice of regularizer is critical as it incorporates the prior assumption about the signal. Some popular ones:
    - l2-norm: Also known as Tikhonov Regularization, ridge regression, weight decay. It is a good choice for communication channels since it minimizes the energy of the signal being set but it is not a good choice as it leads to overly smooth images.
    - Total-variation-norm: It is actually the l1-norm of the gradient and helps the image to retain sharp images. This regularizer works well for MRI images and can be efficiently calculated.
    - Sparse Regularization: When an image is explained in some dictionary(or sparsfying basis like wavelet or fourier), it is more suitable for optimization based approaches. And also total-variation-norm is a special case of sparse regularization, if we consider the sparsifying basis as the finete difference opeator matrix.
- Tikhonov regularization (l2-norm) helps us make the deconvolution operator more stable. Because if the forward operator is ill-conditioned(ratio of singular values goes to infinity) then the noise will be magnified in the case of inverting the forward operator. But with the help of lamda||x||2, we can make the problem well-conditioned but the trade-off here is that we increase the bias of the model. Therefore if we want to decrease the variance of the model by increasing lamda, we also increase the bias of the system. But selecting the lamda a correct value, we can find a better solution than not having it at all.

- min ||Ax-y|| has a closed form solution, by projecting the measurements onto the range of A, by left-multiplying y with the psuedo-inverse(left-inverse) of A;however, this inverse operation is quite hugely costly if the number of rows are huge. Also the solution might not be possible at all. Therefore we are OK with approximated solutions. But since the largeness of the model, we also have to stick with the first-order gradient methods (gradient descent) and hope that the loss-function is convex. Because if the loss-function is convex then we can get a global estimator just by using the first-order derivaties(gradient descent) methods.

- If we analyse the convergence of quadratic loss function with the matrix being positive semi-definite, we end up with the conclusion that N~O(K*log(1/eps)) many iterations will be enough to find a eps-close solution by having a step size of 2/(M+m). Here M is the largest singular value, and m is the smallest singular value of the matrix. And K is the condition number of the matrix which equal to M/m.

- **Proximal Gradient Descent for l1-norm minimization**: The optimization problem min ||x||1 subject to ||Ax-y||2 <= eps, is identical with the optimization problem min ||Ax-y||_2 + lamda||x||_1, but applying gradient descent directly to this optimization problem is not possible as the gradient of l1-norm is not well defined everywhere. Instead we can employ a proximal gradient descent algorithm where we just take the derivative of f(x) = ||Ax-y||_2, which is A.T*A-Ay.
    - solving iteratively x_{k+1} = x_k -eta*grad(f) is identical to solving argmin ||x-(x_k -eta*grad(f))||_2^2 + eta*lamda*||x||_1 because both of them has the premise of having close x's. And the latter minimization problem is coordinate-wise seperable. If we sepeate it into min (x-xi)^2 + eta*lamda*|x|, where xi = [xk-eta*grad(f)]i then this problem has a closed form solution where we use the soft-thresholding function tau, and hence the name of this algorithm, Iterative soft-thresholding algorithm.
    - The algorithm is specifically:
    - Iterative Shrinkage Thresholding Algorithm (ISTA) is a fast iterative for solving the l1-reqularized least-squares optimization problem 

    $$\min_{x} \frac{1}{2} \lVert Ax-y \rVert _{2}^{2}+ \lambda\lVert x \rVert _{1}$$

    for sparse vector reconstruction. ISTA is initilazed with $x_{0}=0$ and its iterations are given by

    $$x_{t+1}=\tau_{\lambda \eta}x_{t}-\eta A^{T}(Ax_{t}-y)$$

    where $\tau_{\eta}$ is the soft-thresholding operator (non-linearity).


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


## **HW6: Noise2Noise vs. Supervised Learning**

- Supervised training requries $(x_{i},y_{i})$ data. A NN $f_{\theta}$ is trained to predict target image based on noisy measuremnt by minimizing the supervised loss:

$$ L(\theta) = \frac{1}{N} \sum_{i=1}^{N} l(f_{\theta} (y_{i}, x_{i}))$$

- But in real-life practice, we don't always have access to the ground-truth, therefore we need different approaches to such problems.

- Three approaches for constructing self-supervised loss:

    - Construct an unbiased estimate of supervised loss

        - requires extra measurements but doesn't require extra assumptions and with sufficent data, gets the performance as in the supervised loss

    - Construct a self-supervised loss based on Stein's unbiased estimator.

    - Construct a self-supervised loss based on predicting one part of an image from another part, and as such makes explicit or implicit assumptions about images. Even with a lot of training data, this approach is typically performs worse than the supervised manner.

### **Self-supervised learning based on estimates of the supervised loss**

- Our goal to get an estimator $f_{\theta}$ which would minimize the supervised risk function on a joint probability density function on data points $(x,y)$
$$R(\boldsymbol{\theta})=\mathbb{E}\left[\ell\left(f_{\boldsymbol{\theta}}(\mathbf{y}), \mathbf{x}\right)\right]$$

    but since we don't have the underlying joint PDF, we use the empirical risk function over N data points

- But we don't have always ground truth; therefore, minimize self-supervised empirical risk function (average loss) over a N measurements and N randomized measurements in expectation over (x,y), minimizing the supervised loss function is the same as minimizing the risk function. Therefore, with sufficient training data we can get the same performance as the supervised learning.

- It is useful in denoising, we need to find $f_{\theta}$ that estimates ground truth images $x$ from the observation $y=x+e$, where e is the additive noise which can be gaussian or non-gaussian. We don't have access the ground truth but we do access the another randomized measurement of the ground truth with another indepent zero mean noise, specifically, $\mathbf{y}_i^{\prime} = \mathbf{x}_i^{\prime} + \mathbf{e}_i^{\prime}$

- Therefore, the self-supervised loss:

$$\ell_{\mathrm{SS}}\left(f_{\boldsymbol{\theta}}(\mathbf{y}), \mathbf{y}^{\prime}\right)=\left\||f_{\boldsymbol{\theta}}(\mathbf{y})-\mathbf{y}^{\prime}\right\||_2^2$$

should be minimized. With enough training data such self-supervised training gives essentially the same performance as super-vised training. This is not surprising given that the supervised loss is an approximation of the risk, and the approximation error goes to zero as the number of training examples N goes to infinity.