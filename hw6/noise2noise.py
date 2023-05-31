## Import of packages
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import math
import random
from unet import Unet

## Build PyTorch Datasets and Dataloaders
dataset_dir = "./hw6/BSDS300"

train_set_dir = f"{dataset_dir}/images/train"
train_img_files = [f"{train_set_dir}/{filename}" for filename in os.listdir(train_set_dir)]
# use this to train with fewer data out of 200 images
Nsamples = 50
train_img_files = random.sample(train_img_files, Nsamples) #304 chunks of training images

test_set_dir = f"{dataset_dir}/images/test"
test_img_files = [f"{test_set_dir}/{filename}" for filename in os.listdir(test_set_dir)]
val_img_files = test_img_files[:50]
test_img_files = test_img_files[50:]

#print(f"Number of selected images {len(train_img_files)}") # outputs 19, because 19 images are randomly selected
#print(f"Selected Images {train_img_files}") # name of the files of the selected 19 images

# load images as grayscale tensors
def load_img(file):
    # open image as grayscale
    img = Image.open(file).convert("L")
    img = torch.tensor(np.array(img))
    # convert to range [0,1]
    img = img / 255.
    return img

# split an image into chunks of size chunk_size x chunks_size (no padding, no overlap)
def chunk_img(img, chunk_size):
    chunks = img.unfold(0, chunk_size, chunk_size).unfold(1, chunk_size, chunk_size).reshape(-1, chunk_size, chunk_size)
    return list(chunks)

# here with the use of "self" keyword we can access the class itself in the class :D
class NoisyImageChunkDataset(Dataset):
    def __init__(self, img_files, noise_var, chunk_size):
        self.img_files = img_files
        self.noise_var = noise_var
        self.chunk_size = chunk_size
        self.chunks_noisy_indp, self.chunk_noisy = self.get_clean_and_noisy_chunks()

    def get_clean_and_noisy_chunks(self):
        # load clean images
        imgs_clean = [load_img(file) for file in self.img_files]
        # split into chunks
        chunks_clean = sum([chunk_img(img, chunk_size=self.chunk_size) for img in imgs_clean], [])
        # add noise to chunks
        chunks_noisy = [img + math.sqrt(self.noise_var) * torch.randn_like(img) for img in chunks_clean]
        # let's create another measurement with independent zero mean, gaussian noise
        chunks_noisy_indp = [img + math.sqrt(self.noise_var) * torch.randn_like(img) for img in chunks_clean]
        return chunks_noisy_indp , chunks_noisy 

    def __len__(self):
        return len(self.chunks_noisy_indp)

    def __getitem__(self, idx):
        return self.chunk_noisy[idx], self.chunks_noisy_indp[idx]
    
noise_var = 0.015  # more noise makes denoising harder, we suggest you keep this value
train_chunk_size = 128  # depends on your hardware; we recommend 128

train_set = NoisyImageChunkDataset(img_files=train_img_files, noise_var=noise_var, chunk_size=train_chunk_size)
# for validation and testing, we do not have to split the images into chunks because we do not have to compute gradients
# the images have shape (321, 481) or (481, 321) so we crop them to (321, 321) to facilitate data loading
val_set = NoisyImageChunkDataset(img_files=val_img_files, noise_var=noise_var, chunk_size=321)
test_set = NoisyImageChunkDataset(img_files=test_img_files, noise_var=noise_var, chunk_size=321)

#plt.imshow(val_set[0][0], cmap="gray")

batch_size = 32  # depends on your hardware

train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# more pooling layers and convolutional kernels increase the complexity of the U-Net (see lecture notes)
num_pool_layers = 2
chans = 64
device = "cpu"  # set to "cuda" or "cuda:0" if you have access to a GPU (e.g. via Google Colab)

model = Unet(
    in_chans=1,  # 1 input channel as we use grayscale images as input
    out_chans=1,  # 1 output channel as the model returns grayscale images
    num_pool_layers=num_pool_layers,
    chans=chans
)
model = model.to(device)

# psnr(img1, img2) = 10 * log10( max_pixel_value^2 / mean_square_error(img1, img2) )
# as we normalized all images to range [0,1], we have that max_pixel_value = 1 and the above formula reduces to
# psnr(img1, img2) = -10 * log10( mean_square_error(img1, img2) )
def get_psnr(gt, pred):
    pred = pred.clamp(0, 1)  # clamp prediction pixel range to range [0,1]
    mse = (gt-pred.clamp(0, 1)).pow(2).mean(dim=(-1,-2))
    return -10 * torch.log10(mse)  # psnr reduces to this formula as we normlaized the images to range [0,1]

def get_training_loss(imgs0, imgs1): #mse loss
    return (imgs0 - imgs1).pow(2).mean(dim=(-1,-2))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # choose a suitable optimizer form torch.optim; we recommend to use the ADAM optimizer

epochs = 10  # how many epochs to train
check_val_every_epochs = 1


for e in range(epochs):
    train_loss = 0.0
    for imgs_noisy, imgs_clean in tqdm.tqdm(train_loader, desc=f"Training Epoch {e}"):
        imgs_noisy = imgs_noisy.to(device)
        imgs_clean = imgs_clean.to(device)
        # clear the gradients
        optimizer.zero_grad()
        # normalize input
        mean = imgs_noisy.mean(dim=(-1, -2), keepdim=True)
        std = imgs_noisy.std(dim=(-1, -2), keepdim=True)
        imgs_noisy = (imgs_noisy - mean) / std
        # forward pass
        imgs_denoised = model(imgs_noisy.unsqueeze(1)).squeeze(1)
        # undo normalizaiton
        imgs_denoised = imgs_denoised * std + mean
        # find the Loss
        loss = get_training_loss(imgs_clean, imgs_denoised).mean()
        # calculate gradients 
        loss.backward()
        # update Weights
        optimizer.step()
        # update epoch loss
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f'Training Loss: {train_loss}')
    
    if e % check_val_every_epochs == 0:
        with torch.no_grad():
            val_psnr = 0.0
            for imgs_noisy, imgs_clean in tqdm.tqdm(val_loader, desc=f"Validation Epoch {e}"):
                imgs_noisy = imgs_noisy.to(device)
                imgs_clean = imgs_clean.to(device)
                mean = imgs_noisy.mean(dim=(-1, -2), keepdim=True)
                std = imgs_noisy.std(dim=(-1, -2), keepdim=True)
                imgs_noisy = (imgs_noisy - mean) / std
                imgs_denoised = model(imgs_noisy.unsqueeze(1)).squeeze(1)
                imgs_denoised = imgs_denoised * std + mean
                psnr = get_psnr(imgs_clean, imgs_denoised).mean()
                val_psnr += psnr.item()
            val_psnr /= len(val_loader)
            print(f'Validation PSNR: {val_psnr}')
            
with torch.no_grad():
    test_psnr = 0.0
    base_psnr = 0.0
    for imgs_noisy, imgs_clean in tqdm.tqdm(test_loader, desc="Testing"):
        imgs_noisy = imgs_noisy.to(device)
        imgs_clean = imgs_clean.to(device)
        mean = imgs_noisy.mean(dim=(-1, -2), keepdim=True)
        std = imgs_noisy.std(dim=(-1, -2), keepdim=True)
        imgs_noisy = (imgs_noisy - mean) / std
        imgs_denoised = model(imgs_noisy.unsqueeze(1)).squeeze(1)
        imgs_denoised = (imgs_denoised * std) + mean
        test_psnr += get_psnr(imgs_clean, imgs_denoised).mean().item()
        base_psnr += get_psnr(imgs_clean, imgs_noisy).mean().item()
    test_psnr /= len(test_loader)
    base_psnr /= len(test_loader)
    print(f"PSNR of noisy images: {base_psnr}")
    print(f'PSNR of denoised images: {test_psnr}')