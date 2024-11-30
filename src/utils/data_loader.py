import random
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


base_path = 'C:/Users/My/Desktop/ActiveLearning_ImageReconstructing/'
patches_path = 'data/patches'


import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
# from get_patches import get_patch_pair_paths


def add_noise(image, noise_typ, noise_level=20, normalized=True):
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
        raise ValueError(f"Unsupported noise type: {noise_typ}, supported types are gaussian, salt_peper, poisson, speckle")

    if not normalized:
        noisy = noisy * 255.0

    return torch.clamp(noisy, 0, 255 if not normalized else 1)

class DenoisingPatchDataset(Dataset):
    def __init__(self, df, transform=None,
                 normalize=None,active_indices=None, 
                 addNoise = False,noise_type=None,noise_level=40
                 ):
        self.addNoise = addNoise
        self.noise_type = noise_type
        self.noise_level = noise_level
        if self.addNoise:
          if self.noise_type is None or self.noise_level is None:
            raise ValueError("If addNoise is True, noise_type and noise_level must be specified.")
          if self.noise_type not in ["gaussian", "salt_pepper", "poisson", "speckle"]:
            raise ValueError("noise_type must be one of 'gaussian', 'salt_pepper', 'poisson', 'speckle'.")
          if self.noise_level < 0 or self.noise_level > 100:
            raise ValueError("noise_level must be between 0 and 100.")
        self.df = df
        self.transform = transform
        self.normalize = normalize
        self.addNoise = addNoise

        if active_indices is not None:
            self.df = self.df.iloc[active_indices]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        clean_path = self.df.iloc[idx]['clean_path']
        noisy_path = self.df.iloc[idx]['noisy_path']

        # Load images as numpy arrays
        clean_patch = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        noisy_patch = cv2.imread(noisy_path, cv2.IMREAD_COLOR)

        # Convert to tensor and apply normalization if specified
        if self.normalize:
            clean_patch = self.normalize(clean_patch)
            noisy_patch = self.normalize(noisy_patch)
            if self.addNoise:
                noisy_patch = add_noise(noisy_patch.unsqueeze(0),self.noise_type,self.noise_level).squeeze(0)
        else:
            clean_patch = torch.from_numpy(clean_patch).permute(2, 0, 1).float()
            noisy_patch = torch.from_numpy(noisy_patch).permute(2, 0, 1).float()
            if self.addNoise:
                noisy_patch = add_noise(noisy_patch.unsqueeze(0),self.noise_type,self.noise_level,normalized=False).squeeze(0)

        # Apply other transforms if defined
        if self.transform:
            clean_patch = self.transform(clean_patch)
            noisy_patch = self.transform(noisy_patch)

        return noisy_patch, clean_patch













def visualize_patches_in_tensor(batch, index=0):
    """Visualize all patches in a single tensor in the batch."""
    # Select the batch tensor to visualize
    tensor = batch
    num_patches = tensor.shape[0]
    
    # Calculate grid size for visualization
    grid_size = int(num_patches ** 0.5)+1
    
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for i in range(num_patches):
        row, col = divmod(i, grid_size)
        patch = TF.to_pil_image(tensor[i])
        axs[row, col].imshow(patch)
        axs[row, col].axis('off')
    
    plt.suptitle(f'All patches from tensor index {index} in the batch')
    plt.show()


# def visualize_corresponding_patches(batch, patch_index=0):
#     """Visualize the patches at the same index across all tensors in the batch."""
#     num_images = len(batch)
    
#     fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
#     for i in range(num_images):
#         patch = TF.to_pil_image(batch[i][patch_index])  # Select the corresponding patch in each image
#         axs[i].imshow(patch)
#         axs[i].set_title(f'Image {i}, Patch {patch_index}')
#         axs[i].axis('off')
    
#     plt.suptitle(f'Patch index {patch_index} from each tensor in the batch')
#     plt.show()







