import numpy as np
import matplotlib.pyplot as plt

def GaussKernel(size, std):    
    x = np.arange(-(size//2), size//2+1)
    kernel = np.exp(-x**2 / (2*std**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def GaussNoise(n, mean, std):
      return np.random.normal(loc=mean, scale=std, size=n)

def A(size,std):
    A = np.zeros((size, size))
    kernel = GaussKernel(size, std)
    for i in range(size):
        A[i, :] = np.roll(kernel, i - size//2)
    return A


n = 101
kernel_std = 2
noise_std = 0.45
noise_mean = 0
granularity = 1000

kernel = GaussKernel(n, kernel_std)

x_star = np.random.random((n, 1))
e = GaussNoise(n,noise_mean,noise_std)
y = A(n,kernel_std) @ x_star + e #noisy observation of x*

U, S, V = np.linalg.svd(A(n,kernel_std))

L = np.linspace(10^-7,10^2,granularity)
bias = np.zeros(granularity)
var = np.zeros(granularity)
mse = np.zeros(granularity)

for j in range(len(L)):
    for i in range(n):
        Si = S[i]
        Vi = V[i]
        lamda = L[j]

        bias[j] += ((1-(Si*Si / (Si*Si + lamda)))**2 )* ((x_star.T @ Vi)**2)
        var[j] += (noise_std**2)*(Si/(Si*Si+lamda))**2
        
mse = bias + var

## PLOTTING ##
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].plot(kernel)
axs[0].set_title('Gaussian Kernel')
axs[1].semilogy(S)
axs[1].set_title('Singular values of A')
axs[2].loglog(L, bias, label='Bias')
axs[2].loglog(L, var, label='Variance')
axs[2].loglog(L, mse, label='MSE')
axs[2].set_title('Bias-Variance Tradeoff')
axs[2].set_xlabel('Lambda')
axs[2].legend()
plt.show()
