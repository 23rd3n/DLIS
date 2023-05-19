# Homework 4: Unet end2end NN training over BSD300 dataset

## Part1: seperating dataset into training, test, and validation:

- **Formatted String** : `f"..."`  
    - Can be used in combination with placeholders {var1}, {var2}
    - Example : 
        - `name = serden`
        - `age = 23`
        - `f"My name is {name} and I'm {age} years old"`

- **List comprehension**:
    - `squared_numbers = [x**2 for x in number_list]` 
    - List comprehension iterates over elements in an iterable (e.g., a list, tuple, or string), applies an expression or transformation to each element, and creates a new list from the results.

- **Indexing**:
    - `my_list[50:]` #elements from index 50 to end
    - `my_list[:50]` #elements until 50
    - `my_list[1:4]`  # Elements from index 1 to 4 (exclusive)
    - `my_list[::2] ` # every second element
    - `my_list[::-1]` # Reverse the list
    - `my_list[1:4] = [10, 20, 30]` #Modify by parts
    - `my_list[1:9:2]` #Elements 1to9 with step size 2
    - `[1,2,3] + [3,4,5] = [1, 2, 3, 3, 4, 5]` #combines two of them into one

## Part2: creating a class for Dataset

- Classes are a fundamental feature of OOP
- allow for reusable and structured code
- provide a blueprint for creating objects with defined behaviors
- by encapsulating data and functionality within a class, we can create multiple instances (objects) of that class, each with its own unique set of attributes and behaviors.
- ```

    class NoisyImageChunkDataset(Dataset):
        def __init__(self, img_files, noise_var, chunk_size):
            self.img_files = img_files
            self.noise_var = noise_var
            self.chunk_size = chunk_size
            self.chunks_clean, self.chunks_noisy = self.get_clean_and_noisy_chunks()

        def get_clean_and_noisy_chunks(self):
        chunks_clean = []
        chunks_noisy = []
        for file in self.img_files:
            
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = img.astype('float32') / 255.0
            height, width = img.shape

            num_chunks_h = height // self.chunk_size
            num_chunks_w = width // self.chunk_size

            # Iterate over chunks
            for i in range(num_chunks_h):
                for j in range(num_chunks_w):
                    # Calculate the chunk coordinates
                    start_h = i * self.chunk_size
                    end_h = start_h + self.chunk_size
                    start_w = j * self.chunk_size
                    end_w = start_w + self.chunk_size

                    # Extract the chunk from the image
                    chunk_clean = img[start_h:end_h, start_w:end_w]
                    # Generate noise for the chunk
                    noise = np.random.normal(0, np.sqrt(self.noise_var), chunk_clean.shape).astype(np.float32)

                    # Add the chunk and noisy version to the lists
                    chunks_clean.append(chunk_clean)
                    chunks_noisy.append(chunk_clean + noise)

        return chunks_clean, chunks_noisy

        def __len__(self):
            return len(self.chunks_clean)

        def __getitem__(self, idx):
            return self.chunks_noisy[idx], self.chunks_clean[idx]

    ```
    - This class **inherits** from the `Dataset` class which is an abstract base Dataset from Pytorch

        - The PyTorch Dataset class is designed to work with large datasets that may not fit entirely into memory. It allows you to load and preprocess data on-the-fly, fetching individual samples when needed, which can be memory-efficient and enable efficient training of deep learning models.

        - By implementing `__getitem__`, `__len__`, and `__init__` methods, your custom dataset class becomes compatible with **PyTorch's data loading** utilities, **such as data loaders and samplers.** These utilities facilitate **efficient loading and batching** of data during training or inference.



    - `__init__` is a method serving as the constructor. The constructor initializes the object's attributes
    
    - `self` is a conventional name used as the first parameter in the method definitions of a class. It is a reference to the instance of the class itself. When you define a method in a class and call it on an object of that class, self is automatically passed as the first argument to the method.

- `img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)` reads a file using OpenCV library.
    - `file` is the path to image you want to read
    - `cv2.IMREAD_GRAYSCALE` is a flag specifying the color mode.

- `plt.imshow(train_set[0][0], cmap="gray")` train_set[0][0] first index specifies the which image to load, and the second index state whether it should display the noisy or clean version of the image. 

- `train_loader = DataLoader(train_set, batch_size=batch_size)` DataLoader is a class provided by PyTorch that helps efficiently loading and batching the data from a dataset. Also allows iterating over the batches. such as `for batch in dataloader:`

## Part 3: Training and Validating using the Dataset

- `model = Unet(in_chans=1,out_chans=1,num_pool_layers=4,chans=64)` is initialize the NN architecture that we used. and `model = model.to("cuda")` moves our model computation into GPU

- `optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5)` ADAM (Adaptive Moment Estimation) is an optimization algortihm commonly used in deep learning. It combines the advantages of AdaGrad and RMSprop to provide efficient and effective updates to the model parameters during the training.

- The data is normalized by calculating the mean and standard deviation of imgs_noisy and subtracting the mean from each element and dividing by the standard deviation. This normalization step helps in stabilizing the training process.

- The optimizer's gradients are reset to zero using `optimizer.zero_grad()`

- The gradients of the loss with respect to the model parameters are computed using `loss.backward()`

- The optimizer takes a step based on the computed gradients to update the model parameters using `optimizer.step()`. This step performs the actual parameter update using the optimization algorithm (Adam in this case).

- PSNR calculated as
    ```
    mse = torch.mean((output - clean) ** 2)
    batch_psnr = 10 * torch.log10(1.0 / mse)
    1.0 used because it is the range of the pixel values
    ```

- **Then the test datasat is used for the verification of the model which was 28.36 dB which is ~8dB more than the noisy versions.**
