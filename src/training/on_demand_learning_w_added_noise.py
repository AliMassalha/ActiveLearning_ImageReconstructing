import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import sys
import os
import gc
import torch.cuda
from tqdm import tqdm
import numpy as np
import cv2
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import DenoisingPatchDataset, add_noise

# from utils.combine_data_and_patchify import combining_data
from utils.get_patches import get_patch_pair_paths
from models.unet import LightDownBlock, LightUpBlock, LightDenoisingUNet
from models.ridnet import LightEAM, LightRIDNet

def estimate_noise_level(noisy_img, clean_img):
    """Estimate noise level by computing PSNR between noisy and clean images"""
    mse = np.mean((noisy_img - clean_img) ** 2)
    noise_std = np.sqrt(mse)
    return noise_std

def add_noise(noise_typ, image, noise_level=20, normalized=True):
    if not normalized:
        image = image / 255.0

    BS, ch, row, col = image.shape
    noise_factor = noise_level / 100.0

    if noise_typ == "gaussian":
        mean = 0
        var = 0.1 * noise_factor
        sigma = var**0.5
        gauss = torch.randn_like(image,device=image.device) * sigma
        noisy = image + gauss
    elif noise_typ == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.004 * noise_factor
        out = image.clone()
        num_salt = int(amount * image.numel() * s_vs_p)
        salt_coords = [torch.randint(0, dim, (num_salt,)) for dim in image.shape]
        out[salt_coords] = 1
        num_pepper = int(amount * image.numel() * (1 - s_vs_p))
        pepper_coords = [torch.randint(0, dim, (num_pepper,)) for dim in image.shape]
        out[pepper_coords] = 0
        noisy = out
    elif noise_typ == "poisson":
        vals = len(torch.unique(image))
        vals = 2 ** torch.ceil(torch.log2(torch.tensor(vals, dtype=torch.float)))
        noisy = torch.poisson(image * vals * noise_factor) / vals
    elif noise_typ == "speckle":
        gauss = torch.randn_like(image) * noise_factor
        noisy = image + image * gauss
    else:
        raise ValueError(f"Unsupported noise type: {noise_typ}")

    if not normalized:
        noisy = noisy * 255.0

    return torch.clamp(noisy, 0, 255 if not normalized else 1)


class NoiseGroupedDenoisingDataset(Dataset):
    def __init__(self, df, transform=None, normalize=None, group=False):
        self.df = df
        self.transform = transform
        self.normalize = normalize
        self.noise_levels = []
        self.noise_groups = []
        if group:
          self.setup_noise_groups()

    def setup_noise_groups(self):
        """Estimate noise levels and group images"""
        print("Analyzing noise levels in dataset...")
        noise_levels = []

        for idx in range(len(self.df)):
            clean_path = self.df.iloc[idx]['clean_path']
            noisy_path = self.df.iloc[idx]['noisy_path']

            clean_img = cv2.imread(clean_path, cv2.IMREAD_COLOR)
            noisy_img = cv2.imread(noisy_path, cv2.IMREAD_COLOR)

            noise_level = estimate_noise_level(noisy_img, clean_img)
            noise_levels.append(noise_level)

        self.noise_levels = np.array(noise_levels)

        

        # percentiles = np.percentile(self.noise_levels, [50])
        # self.noise_groups = [
        #     np.where(self.noise_levels < percentiles[0])[0],
        #     np.where(self.noise_levels >= percentiles[0])[0],
        # ]

        percentiles = np.percentile(self.noise_levels, [33,66])
        self.noise_groups = [
            np.where(self.noise_levels < percentiles[0])[0],
            np.where((self.noise_levels >= percentiles[0]) & (self.noise_levels < percentiles[1]))[0],
            np.where(self.noise_levels >= percentiles[1])[0],
        ]
        # Create 5 groups based on noise levels
        # percentiles = np.percentile(self.noise_levels, [20, 40, 60, 80])
        # self.noise_groups = [
        #     np.where(self.noise_levels < percentiles[0])[0],
        #     np.where((self.noise_levels >= percentiles[0]) & (self.noise_levels < percentiles[1]))[0],
        #     np.where((self.noise_levels >= percentiles[1]) & (self.noise_levels < percentiles[2]))[0],
        #     np.where((self.noise_levels >= percentiles[2]) & (self.noise_levels < percentiles[3]))[0],
        #     np.where(self.noise_levels >= percentiles[3])[0]
        # ]
        
        print(f"Noise level ranges for groups:")
        for i, percentile in enumerate(percentiles):
            print(f"Group {i}: {percentile:.2f}")

    def get_group_indices(self, group_idx):
        """Get indices for a specific noise level group"""
        return self.noise_groups[group_idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        clean_path = self.df.iloc[idx]['clean_path']
        noisy_path = self.df.iloc[idx]['noisy_path']

        clean_patch = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        noisy_patch = cv2.imread(noisy_path, cv2.IMREAD_COLOR)

        if self.normalize:
            clean_patch = self.normalize(clean_patch)
            noisy_patch = self.normalize(noisy_patch)
        else:
            clean_patch = torch.from_numpy(clean_patch).permute(2, 0, 1).float()
            noisy_patch = torch.from_numpy(noisy_patch).permute(2, 0, 1).float()

        # Apply other transforms if defined
        if self.transform:
            clean_patch = self.transform(clean_patch)
            noisy_patch = self.transform(noisy_patch)

        return noisy_patch, clean_patch
class Options:
    def __init__(self):
        self.batchSize = 60
        self.fine_size = 256
        self.nef = 32  # encoder filters
        self.ngf = 32  # generator filters
        self.nc = 3    # channels
        self.lr = 0.001 #0.0002
        self.beta1 = 0.5
        self.nThreads = 0 ##########change to zero
        self.manualSeed = 0
        self.n_epochs = 15
        self.gpu = True if torch.cuda.is_available() else False
        self.name = 'denoise'

        self.noise_types = ['gaussian',  'speckle']#'salt_pepper', 'poisson',
        self.noise_levels = [20, 40, 60, 80, 100]
        self.subtasks = [(noise_type, noise_level) for noise_type in self.noise_types for noise_level in self.noise_levels]
        # self.subtasks = [20, 40, 60, 80, 100]  # Noise levels

        self.batchSize_per_subtask = [self.batchSize // len(self.subtasks)] * len(self.subtasks)
        # On-demand learning parameters
        self.N = len(self.subtasks)  # number of noise level groups
        self.B = self.batchSize  # total batch size
        self.display_iter = 50

opt = Options()

# Instantiate model, optimizer, and loss
device = torch.device("cuda" if opt.gpu > 0 and torch.cuda.is_available() else "cpu")
# model = Generator1(opt.nef, opt.ngf, opt.nc).to(device)
model = LightRIDNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#, betas=(opt.beta1, 0.999)

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

patches_dir = 'data/patches'
df = get_patch_pair_paths(patches_dir)
df_sample = df.sample(frac=0.25,random_state=42)
train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42, shuffle=True)
train_dataset = NoiseGroupedDenoisingDataset(train_df, normalize=normalize_transform, group=False)
validation_dataset = NoiseGroupedDenoisingDataset(test_df, normalize=normalize_transform, group=False)
if train_dataset.normalize:
  max_pixel = 1.0
else:
  max_pixel = 255.0
train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads)
validation_loader = DataLoader(validation_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.nThreads)



# On-demand learning: Training loop with dynamic reallocation
subtask_psnr = [0] * len(opt.subtasks)  # Initialize PSNR scores
for epoch in range(opt.n_epochs):
    print(f"Epoch {epoch+1}/{opt.n_epochs}")

    # Training loop
    for i, (noisy_images, clean_images) in enumerate(train_loader):
        clean_images = clean_images.to(device)
        noisy_images = noisy_images.to(device)
        actual_batch_size = clean_images.size(0)

        # Shuffle clean_images at the start of each epoch to ensure randomness
        rand_perm = torch.randperm(actual_batch_size)
        clean_images = clean_images[rand_perm]
        noisy_images = noisy_images[rand_perm]

        corrupted_batches = []
        clean_batches = []

        # Scale the allocations to match the actual batch size
        total_allocated = sum(opt.batchSize_per_subtask)
        scaled_allocations = [int(round(size * actual_batch_size / total_allocated)) for size in opt.batchSize_per_subtask]

        # Adjust for rounding errors to match batch size exactly
        while sum(scaled_allocations) != actual_batch_size:
            if sum(scaled_allocations) < actual_batch_size:
                # Add the remaining samples to the subtask with the highest allocation
                max_idx = scaled_allocations.index(max(scaled_allocations))
                scaled_allocations[max_idx] += 1
            else:
                # Remove samples from the subtask with the highest allocation
                max_idx = scaled_allocations.index(max(scaled_allocations))
                scaled_allocations[max_idx] -= 1

        # Divide the batch across sub-tasks using scaled allocations
        start_idx = 0
        for idx, ((noise_type, noise_level), num_samples) in enumerate(zip(opt.subtasks, scaled_allocations)):
            if num_samples == 0:
                continue

            end_idx = start_idx + num_samples

            # Select a slice of the shuffled batch
            clean_batch = clean_images[start_idx:end_idx]
            noise_batch = noisy_images[start_idx:end_idx]

            # Add noise and save
            corrupted_batches.append(add_noise(noise_type, noise_batch, noise_level=noise_level))
            clean_batches.append(clean_batch)

            start_idx = end_idx

        # Combine all batches
        corrupted_images = torch.cat(corrupted_batches, dim=0)
        clean_images = torch.cat(clean_batches, dim=0)

        # Forward pass
        optimizer.zero_grad()
        restored_images = model(corrupted_images)
        loss = criterion(restored_images, clean_images)
        loss.backward()
        optimizer.step()

        # Display progress
        if i % opt.display_iter == 0:
            print(f"Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
            print(f"Batch allocations: {scaled_allocations}")

    # Validation phase to update subtask proportions
    validation_psnr = [0] * len(opt.subtasks)
    subtask_counts = [0] * len(opt.subtasks)

    with torch.no_grad():
        for noisy_images, clean_images in validation_loader:
            clean_images = clean_images.to(device)
            noisy_images = noisy_images.to(device)

            for idx, (noise_type, noise_level) in enumerate(opt.subtasks):
                num_samples = clean_images.size(0)
                noise = add_noise(noise_type, noisy_images, noise_level=noise_level)
                restored = model(noise)

                mse = np.mean((clean_images.cpu().numpy() - restored.cpu().numpy()) ** 2)
                psnr = 10 * np.log10(max_pixel**2 / mse)

                validation_psnr[idx] += psnr * num_samples
                subtask_counts[idx] += num_samples

    # Average PSNR for each sub-task
    validation_psnr = [psnr / count for psnr, count in zip(validation_psnr, subtask_counts)]
    print(f"Validation PSNR: {validation_psnr}")

    # Update allocation based on inverse PSNR
    inverse_psnr = [1 / max(psnr, 1e-10) for psnr in validation_psnr]
    total_inverse_psnr = sum(inverse_psnr)
    opt.batchSize_per_subtask = [(1 / psnr) / total_inverse_psnr * opt.batchSize for psnr in validation_psnr]
    print(f"New batch distribution: {opt.batchSize_per_subtask}")

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')


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

    print(f"\nResults:")
    print(f"Mean PSNR of noisy images: {mean_psnr_noisy:.2f} dB")
    print(f"Mean PSNR of denoised images: {mean_psnr_denoised:.2f} dB")
    print(f"PSNR improvement: {mean_psnr_denoised - mean_psnr_noisy:.2f} dB")

    return mean_psnr_noisy, mean_psnr_denoised


evaluate_psnr(model, validation_loader, device, max_pixel)