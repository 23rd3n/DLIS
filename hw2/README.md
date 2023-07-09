## This homework, as investigated in [README](../README.md), just an implementation of ISTA algorithm to see if it is able to reconstruct a sparse signal and to plot its convergence as the number of iterations. Here, I've listed some of the things that I don't know but rest of the code is crystal-clear.

### What is tqdm in python? Let's dive deeper:
- tqdm means "progress" in Arabic (taqadum, تقدّم) and is an abbreviation for "I love you so much" in Spanish (te quiero demasiado).
- Makes your loops a smart progress bar. Just wrap any iterable with tqdm(iterable), and it will show progress bar.
- `from tqdm import tqdm` --> `for i in tqdm(range(number_of_iters))` will be enough to create a progress bar.

### What is an iterable in python?:
- An iterable is any Python object capable of returning its members one at a time, permitting it to be iterated over in a for-loop.
- List, tuple, dict, set: construct a list, tuple, dictionary, or set, respectively, from the contents of an iterable

### Sparsify a vector? How to do it efficiently:
```Python
#sparsify x
zero_indices = np.random.choice(np.arange(1000),replace=False,size=int(900)) #900 random indices
x[zero_indices]=0 #x is 100-sparse
```
- `choice`: allows you to sample elements from a given array
    - `replace=False`: selects uniquely, no number will be selected more than once
    - `arange(1000)`: creates an array of numbers from 0 to 999
    - `size=int(900)`: select 900 numbers

- `np.linalg.norm(A@x-y)**2` : There is also euclidian norm defined in numpy library.