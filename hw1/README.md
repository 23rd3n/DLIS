- `iterations = np.linspace(0,n,num=11).astype(int)` will allow us to create different values of k in an iterable way, so that we can analyze our denoising performance for a different number of subspace input sizes.
- The following two lines of code is a way of creating a subspace in python. The first line will return singular value decomposition matrices (USV') for an input n-by-n input matrix but since we only want to subspace, we take the U matrix
```Python
OM = np.linalg.svd(np.random.rand(n,n))
OM = OM[0]
```
- `P = np.random.rand(n,500)` 500 random points of dimension n
- Then for different sizes of k, we will do:
```Python
# Orthonormal basis of the subspace, size nxk
U = OM[:,0:k] # take first k columns

# Project random points on the subspace to make the randomly generated points to lie in the same subspace
# Note that here, we do not work with (U @ U.T) @ P directly, 
# because U @ (U.T @ P) is computationally more efficient 
X = U @ (U.T @ P)

# Gaussian noise with mean 0, variance 1
Z = np.random.normal(0, 1, size=(n,500))

# Noisy observations
Y = X + Z

# Denoise, peoject the noisy obersvations also the subspace
X_hat = U @ (U.T @ Y)

#average MSE 
mse.append(np.average(np.sum((X-X_hat)**2,0)/np.sum(X**2,0)))

#std of MSE
std.append(np.std(np.sum((X-X_hat)**2,0)/np.sum(X**2,0)))
```