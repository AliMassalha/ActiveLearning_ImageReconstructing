import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import os
import gc
import torch.cuda
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import DenoisingPatchDataset, add_noise

# from utils.combine_data_and_patchify import combining_data
from utils.get_patches import get_patch_pair_paths

from models.unet import LightDownBlock, LightUpBlock, LightDenoisingUNet
from models.ridnet import LightEAM, LightRIDNet

def set_global_seeds(random_seed=1234):
        """Set seeds for all random number generators."""
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
set_global_seeds()

patches_dir = 'data/patches'
df = get_patch_pair_paths(patches_dir)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and split data
print("Loading data...")
df_sample = df.sample(frac=0.2,random_state=42)
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42, shuffle=True)

normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

augment_transform = transforms.Compose([
    # transforms.RandomCrop(224, 224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),

])

batch_size = 64

train_dataset = DenoisingPatchDataset(train_df, normalize=normalize_transform, transform=None, active_indices=None)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)

test_dataset = DenoisingPatchDataset(test_df, normalize=normalize_transform, transform=None, active_indices=None)

test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)

if train_dataset.normalize:
  max_pixel = 1.0
else:
  max_pixel = 255.0

print(f"Max pixel value: {max_pixel}")





def train_step(model, optimizer, criterion, noisy_batch, clean_batch, device):

        model.train()
        optimizer.zero_grad()

        # Forward pass
        denoised = model(noisy_batch)

        # Calculate loss
        loss = criterion(denoised, clean_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Clear cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return loss.item()


def validate(model, val_loader, criterion, device,max_pixel=1.0):
    model.eval()
    total_loss = 0
    num_batches = 0
    total_psnr_noisy = 0
    total_psnr_denoised = 0
    num_b = len(val_loader)

    def calculate_psnr(img1, img2, max_val):
        # Ensure the images are in range [0, 1]
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = max_val
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    with torch.no_grad():
        for noisy_batch, clean_batch in val_loader:
            noisy_batch = noisy_batch.float().to(device)
            clean_batch = clean_batch.float().to(device)

            denoised = model(noisy_batch)
            loss = criterion(denoised, clean_batch)
            total_loss += loss.item()
            num_batches += 1
            # Calculate PSNR for each image in batch
            for i in range(clean_batch.size(0)):
                # Original noisy image PSNR
                psnr_noisy = calculate_psnr(clean_batch[i], noisy_batch[i],max_pixel)
                total_psnr_noisy += psnr_noisy

                # Denoised image PSNR
                psnr_denoised = calculate_psnr(clean_batch[i], denoised[i],max_pixel)
                total_psnr_denoised += psnr_denoised

    # Calculate means
    mean_psnr_noisy = total_psnr_noisy / (num_b * val_loader.batch_size)
    mean_psnr_denoised = total_psnr_denoised / (num_b * val_loader.batch_size)



    return total_loss / num_batches , mean_psnr_noisy, mean_psnr_denoised




@torch.no_grad()
def evaluate_psnr(model, dataloader, device, max_val=255.0):
    model.eval()
    total_psnr_noisy = 0
    total_psnr_denoised = 0
    num_batches = len(dataloader)

    def calculate_psnr(img1, img2, max_val):
        # Ensure the images are in range [0, 1]
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = max_val
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    for clean, noisy in tqdm(dataloader, desc="Calculating PSNR"):
        clean = clean.to(device)
        noisy = noisy.to(device)
        # print(clean.max(), clean.min())
        # Get model prediction
        denoised = model(noisy).to(device)

        # Calculate PSNR for each image in batch
        for i in range(clean.size(0)):
            # Original noisy image PSNR
            psnr_noisy = calculate_psnr(clean[i], noisy[i],max_val)
            total_psnr_noisy += psnr_noisy

            # Denoised image PSNR
            psnr_denoised = calculate_psnr(clean[i], denoised[i],max_val)
            total_psnr_denoised += psnr_denoised

    # Calculate means
    mean_psnr_noisy = total_psnr_noisy / (num_batches * dataloader.batch_size)
    mean_psnr_denoised = total_psnr_denoised / (num_batches * dataloader.batch_size)


    return mean_psnr_noisy, mean_psnr_denoised






# Initialize model and training components
print("Initializing model...")

# model = LightDenoisingUNet().to(device)
model = LightRIDNet().to(device)



criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15
print("Starting training...")


mse_scores = []
psnr_improvment_scores = []
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    # Use tqdm for progress bar
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (noisy_batch, clean_batch) in enumerate(train_iterator):

            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            loss = train_step(model, optimizer, criterion, noisy_batch, clean_batch, device)
            epoch_loss += loss
            num_batches += 1
            train_iterator.set_postfix({'batch_loss': f'{loss:.6f}'})



    # Validation
    val_loss,mean_psnr_noisy,mean_psnr_denoised = validate(model, test_loader, criterion, device)

    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {epoch_loss/num_batches:.6f}')
    print(f'Validation Loss: {val_loss:.6f}\n')
    
    print(f"Mean PSNR of noisy images: {mean_psnr_noisy:.2f} dB")
    print(f"Mean PSNR of denoised images: {mean_psnr_denoised:.2f} dB")
    print(f"PSNR improvement: {mean_psnr_denoised - mean_psnr_noisy:.2f} dB")

 
    mse_scores.append(val_loss)
    psnr_improvment_scores.append(mean_psnr_denoised - mean_psnr_noisy)

print(f"MSE scores: {mse_scores}")
print(f"PSNR improvement scores: {psnr_improvment_scores}")


try:
    PATH = 'trained_ridnet_model.pth'
    torch.save(model.state_dict(), PATH)
    print('Model saved successfully')
except Exception as e:
    PATH = os.path.join(os.getcwd(), 'trained_unet_model.pth')
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print(f'Model saved to alternate path: {PATH}')


# Initialize model and training components
print("Initializing model...")

model = LightDenoisingUNet().to(device)

# model = LightRIDNet().to(device)




criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15
print("Starting training...")


mse_scores = []
psnr_improvment_scores = []
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    # Use tqdm for progress bar
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (noisy_batch, clean_batch) in enumerate(train_iterator):

            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            loss = train_step(model, optimizer, criterion, noisy_batch, clean_batch, device)
            epoch_loss += loss
            num_batches += 1
            train_iterator.set_postfix({'batch_loss': f'{loss:.6f}'})



    # Validation
    val_loss,mean_psnr_noisy,mean_psnr_denoised = validate(model, test_loader, criterion, device)

    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {epoch_loss/num_batches:.6f}')
    print(f'Validation Loss: {val_loss:.6f}\n')
   
    print(f"Mean PSNR of noisy images: {mean_psnr_noisy:.2f} dB")
    print(f"Mean PSNR of denoised images: {mean_psnr_denoised:.2f} dB")
    print(f"PSNR improvement: {mean_psnr_denoised - mean_psnr_noisy:.2f} dB")


    mse_scores.append(val_loss)
    psnr_improvment_scores.append(mean_psnr_denoised - mean_psnr_noisy)

print(f"MSE scores: {mse_scores}")
print(f"PSNR improvement scores: {psnr_improvment_scores}")


try:
    PATH = 'trained_unet_model.pth'
    torch.save(model.state_dict(), PATH)
    print('Model saved successfully')
except Exception as e:
    PATH = os.path.join(os.getcwd(), 'trained_unet_model.pth')
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print(f'Model saved to alternate path: {PATH}')



