# SERDEN SAIT ERANIL

import os
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm  # for nice progress bars
from matplotlib import pyplot as plt
import cv2
import numpy as np


from unet import Unet



dataset_dir = "./BSDS300"

train_set_dir = f"{dataset_dir}/images/train"
train_img_files = [f"{train_set_dir}/{filename}" for filename in os.listdir(train_set_dir)]

test_set_dir = f"{dataset_dir}/images/test"
test_img_files = [f"{test_set_dir}/{filename}" for filename in os.listdir(test_set_dir)]
val_img_files = test_img_files[:50]
test_img_files = test_img_files[50:]


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
          #print(f"image shape = {img.shape}")
          img = img.astype('float32') / 255.0
          height, width = img.shape

          # Calculate the number of chunks in each dimension
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



noise_var = 0.005  # more noise makes denoising harder; we suggest you keep this value but you can also experiment with more or less noise
train_chunk_size = 128  # depends on your hardware; larger chunks require more memory during gradient computation; we recommend 128

train_set = NoisyImageChunkDataset(img_files=train_img_files, noise_var=noise_var, chunk_size=train_chunk_size)
# for validation and testing, we do not have to split the images into chunks because we do not have to compute gradients
# the images have shape (321, 481) or (481, 321) so we crop them to (321, 321) to facilitate data loading
val_set = NoisyImageChunkDataset(img_files=val_img_files, noise_var=noise_var, chunk_size=321)
test_set = NoisyImageChunkDataset(img_files=test_img_files, noise_var=noise_var, chunk_size=321)

plt.imshow(train_set[0][0], cmap="gray")

print(train_set[8][1].shape)

batch_size = 1  # depends on your hardware

train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

print(len(train_loader))

# more pooling layers and convolutional kernels increase the complexity of the U-Net (see lecture notes)
num_pool_layers = 4
chans = 64
device = "cuda"  # set to "cuda" or "cuda:0" if you have access to a GPU (e.g. via Google Colab)

model = Unet(
    in_chans=1,  # 1 input channel as we use grayscale images as input
    out_chans=1,  # 1 output channel as the model returns grayscale images
    num_pool_layers=num_pool_layers,
    chans=chans
)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5)  # choose a suitable optimizer form torch.optim; we recommend to use the ADAM optimizer

epochs = 5  # how many epochs to train

check_val_every_epochs = 5

# Define the loss function
criterion = torch.nn.MSELoss()


for e in range(epochs):
    for imgs_noisy, imgs_clean in tqdm.tqdm(train_loader, desc="Training"): #
        imgs_noisy = imgs_noisy.to(device)
        imgs_clean = imgs_clean.to(device)

        imgs_noisy = imgs_noisy.unsqueeze(1)
        imgs_clean = imgs_clean.unsqueeze(1)
        
        # Normalize the data
        mean = imgs_noisy.mean()
        std = imgs_noisy.std()
        imgs_noisy_norm = (imgs_noisy - mean) / std
        #print(imgs_noisy.shape)
        #imgs_noisy_norm  = imgs_noisy_norm .unsqueeze(1)
        outputs = model(imgs_noisy_norm)

        #De-normalize the model output
        outputs = outputs * std + mean

        loss = criterion(outputs, imgs_clean)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    if e % check_val_every_epochs == 0:
        # Disable gradient computation for validation
        with torch.no_grad():
            total_val_loss = 0.0
            total_psnr = 0.0
            num_val_samples = 0

            for imgs_noisy, imgs_clean in tqdm.tqdm(val_loader, desc="Validation"):
                imgs_noisy = imgs_noisy.to(device)
                imgs_clean = imgs_clean.to(device)

                imgs_noisy = imgs_noisy.unsqueeze(1)
                imgs_clean = imgs_clean.unsqueeze(1)

                # Normalize the data
                imgs_noisy_norm = (imgs_noisy - mean) / std

                # Forward pass
                #imgs_noisy_norm = imgs_noisy_norm.unsqueeze(1)
                outputs = model(imgs_noisy_norm)

                # De-normalize the model output
                outputs = outputs * std + mean

                # Compute the validation loss
                val_loss = criterion(outputs, imgs_clean)
                total_val_loss += val_loss.item()

                # Compute PSNR (Peak Signal-to-Noise Ratio) as a metric
                mse = torch.mean((outputs - imgs_clean) ** 2)
                psnr = 10 * torch.log10(1.0 / mse)
                total_psnr += psnr.item()

                num_val_samples += imgs_noisy.size(0)

            avg_val_loss = total_val_loss / num_val_samples
            avg_psnr = total_psnr / num_val_samples

            print(f"Epoch: {e}, Validation Loss: {avg_val_loss}, PSNR: {avg_psnr}")



class ImageMetrics:
    def __init__(self):
        self.total_psnr = 0.0
        self.total_ssim = 0.0
        self.num_samples = 0

    def update(self, output, clean):
        mse = torch.mean((output - clean) ** 2)
        batch_psnr = 10 * torch.log10(1.0 / mse)
        self.total_psnr += batch_psnr
        self.num_samples += output.shape[0]

    def get_metrics(self):
        avg_psnr = self.total_psnr / self.num_samples
        return avg_psnr

metrics = ImageMetrics()

with torch.no_grad():
    for imgs_noisy, imgs_clean in tqdm.tqdm(test_loader, desc="Testing"):
        imgs_noisy = imgs_noisy.to(device)
        imgs_clean = imgs_clean.to(device)

        imgs_noisy = imgs_noisy.unsqueeze(1)
        imgs_clean = imgs_clean.unsqueeze(1)

        # Normalize the data
        mean = imgs_noisy.mean()
        std = imgs_noisy.std()
        imgs_noisy_norm = (imgs_noisy - mean) / std

        # Forward pass
        outputs = model(imgs_noisy_norm)

        # De-normalize the model output
        outputs = outputs * std + mean

        metrics.update(outputs, imgs_clean)

# Get the average PSNR and SSIM
avg_psnr = metrics.get_metrics()

print(f"Average PSNR: {avg_psnr:.2f}")

# **Average PSNR: 28.36**

plt.imshow(outputs[0][0].cpu().numpy(), cmap="gray")


plt.imshow(test_set[49][0], cmap="gray")