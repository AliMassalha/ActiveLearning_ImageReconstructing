import glob
import os
import random
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

base_path = 'C:/Users/My/Desktop/ActiveLearning_ImageReconstructing/'
sidd_path = 'data/SIDD_Medium_Srgb/Data'
renoir_path = 'data/RENOIR'
polyu_path = 'data/PolyU_PairedImages'

# List to store paths for each dataset
data = []

# 1. Gather SIDD "pair1" data
for folder in os.listdir(sidd_path):
    if "pair1" in folder:
        folder_path = os.path.join(sidd_path, folder)
        gt_image = glob.glob(os.path.join(folder_path, '*GT*.PNG'))[0]
        noisy_image = glob.glob(os.path.join(folder_path, '*NOISY*.PNG'))[0]
        data.append({'dataset': 'SIDD', 'gt_path': gt_image, 'noisy_path': noisy_image})

# 2. Gather RENOIR data
camera_types = ['Mi3_Aligned', 'T3i_Aligned', 'S90_Aligned']
for camera in camera_types:
    camera_path = os.path.join(renoir_path, camera)
    for batch_folder in os.listdir(camera_path):
        batch_path = os.path.join(camera_path, batch_folder)
        if os.path.isdir(batch_path):
            reference_files = glob.glob(os.path.join(batch_path, '*Reference.bmp'))
            noisy_files = glob.glob(os.path.join(batch_path, '*Noisy.bmp'))
            for ref, noisy in zip(reference_files, noisy_files):
                data.append({'dataset': 'RENOIR', 'gt_path': ref, 'noisy_path': noisy})


# 3. Gather PolyU data
for folder in os.listdir(polyu_path):
    folder_path = os.path.join(polyu_path, folder)
    for pair in os.listdir(folder_path):
        pair_path = os.path.join(folder_path, pair)
        gt_image = glob.glob(os.path.join(pair_path, '*mean*'))[0]
        noisy_image = glob.glob(os.path.join(pair_path, '*noisy*'))[0]
        data.append({'dataset': 'PolyU', 'gt_path': gt_image, 'noisy_path': noisy_image})

df = pd.DataFrame(data)



class DenoisingDataset(Dataset):
    def __init__(self, df, patch_size=128, stride=128, num_random_patches=10, 
                 augment=False, normalize=True):
        """
        Data loader for image denoising tasks.
        
        Args:
            df (pd.DataFrame): DataFrame with image pairs.
            patch_size (int): Size of each patch.
            stride (int): Stride for patch extraction.
            num_random_patches (int): Number of random patches to extract.
            patchify (bool): Whether to use patchification.
            augment (bool): Whether to apply data augmentation.
            normalize (bool): Whether to normalize images to [0,1].
        """
        self.df = df
        self.patch_size = patch_size
        self.stride = stride
        self.num_random_patches = num_random_patches
        # self.patchify = patchify
        self.augment = augment
        self.normalize = normalize
        self.transform = self._get_transform()

    def _get_transform(self):
        """Set up augmentations and normalization if needed."""
        transform_list = []
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15)
            ])
        if self.normalize:
            transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.df)

    def _extract_patches(self, image):
        """Extract patches from an image based on patch size and stride."""
        patches = []
        for y in range(0, image.shape[0] - self.patch_size + 1, self.stride):
            for x in range(0, image.shape[1] - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
        return patches

    def _get_patch_indices(self, num_patches):
        """Generate random indices for patches if needed."""
        if num_patches <= self.num_random_patches:
            return list(range(num_patches))  # All patches if fewer than needed
        return random.sample(range(num_patches), self.num_random_patches)

    # def _get_patches(self, img, random_patch=False):
    #     """Get either random patches or whole image patches."""
    #     if self.patchify:
    #         patches = self._extract_patches(img)
    #         if random_patch:
    #             num_patches_to_sample = min(self.num_random_patches, len(patches))
    #             patches = random.sample(patches, num_patches_to_sample)
    #         return patches
    #     return [img]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gt_img = np.array(Image.open(row['gt_path']).convert('RGB'))
        noisy_img = np.array(Image.open(row['noisy_path']).convert('RGB'))
        
        # Extract patches once for both images
        all_gt_patches = self._extract_patches(gt_img)
        all_noisy_patches = self._extract_patches(noisy_img)

        if self.num_random_patches and len(all_gt_patches) > self.num_random_patches:
            indices = self._get_patch_indices(len(all_gt_patches))
            gt_patches = [all_gt_patches[i] for i in indices]
            noisy_patches = [all_noisy_patches[i] for i in indices]
        else:
            gt_patches = all_gt_patches
            noisy_patches = all_noisy_patches

        if self.transform:
            gt_patches = [self.transform(Image.fromarray(patch)) for patch in gt_patches]
            noisy_patches = [self.transform(Image.fromarray(patch)) for patch in noisy_patches]

        return gt_patches, noisy_patches

        
    

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

batch_size = 32
patch_size = 128
stride = 128
num_random_patches = 10
# Train Dataset and DataLoader
train_dataset = DenoisingDataset(train_df, patch_size=patch_size, stride=stride, 
                                 num_random_patches=num_random_patches,  
                                 augment=False, normalize=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Test Dataset and DataLoader
test_dataset = DenoisingDataset(test_df, patch_size=patch_size, stride=stride, 
                                num_random_patches=num_random_patches, 
                                augment=False, normalize=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)







def visualize_patches_in_tensor(batch, index=0):
    """Visualize all 32 patches in a single tensor in the batch."""
    # Select the batch tensor to visualize
    tensor = batch[index]  # (32, 3, 128, 128)
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


def visualize_corresponding_patches(batch, patch_index=0):
    """Visualize the patches at the same index across all tensors in the batch."""
    num_images = len(batch)
    
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        patch = TF.to_pil_image(batch[i][patch_index])  # Select the corresponding patch in each image
        axs[i].imshow(patch)
        axs[i].set_title(f'Image {i}, Patch {patch_index}')
        axs[i].axis('off')
    
    plt.suptitle(f'Patch index {patch_index} from each tensor in the batch')
    plt.show()


# Example: Visualize a single batch
print('train_loader: ',len(train_loader))
for gt_batch, noisy_batch in train_loader:
    try:
        print(f"GT Batch Shape: {gt_batch.shape}")
        print(f"GT Batch type: {type(gt_batch)}")
        print(f"Noisy Batch Shape: {noisy_batch.shape}")
        print(f"Noisy Batch type: {type(noisy_batch)}")
    except:
        print(f"GT Batch type: {type(gt_batch)}")
        print(f"GT Batch Shape: {len(gt_batch)}")
        print(f"GT Batch member type: {type(gt_batch[0])}")
        print(f"GT Batch member shape: {gt_batch[0].shape}")

        print(f"Noisy Batch type: {type(noisy_batch)}")
        print(f"Noisy Batch Shape: {len(noisy_batch)}")
        print(f"Noisy Batch member type: {type(noisy_batch[0])}")
        print(f"Noisy Batch member shape: {noisy_batch[0].shape}")
    
    # visualize_patches_in_tensor(gt_batch, index=0)  # visualize all patches in the first tensor of gt_batch
    # visualize_patches_in_tensor(noisy_batch, index=0)  
    
    visualize_corresponding_patches(gt_batch, patch_index=0)  # visualize patch 0 across all images in gt_batch
    visualize_corresponding_patches(noisy_batch, patch_index=0)
    
    break



# print('test_loader: ',len(test_loader))
# for gt_batch, noisy_batch in test_loader:
#     try:
#         print(f"GT Batch Shape: {gt_batch.shape}")
#         print(f"GT Batch type: {type(gt_batch)}")
#         print(f"Noisy Batch Shape: {noisy_batch.shape}")
#         print(f"Noisy Batch type: {type(noisy_batch)}")
#     except:
#         print(f"GT Batch type: {type(gt_batch)}")
#         print(f"GT Batch Shape: {len(gt_batch)}")
#         print(f"GT Batch member type: {type(gt_batch[0])}")
#         print(f"GT Batch member shape: {gt_batch[0].shape}")

#         print(f"Noisy Batch type: {type(noisy_batch)}")
#         print(f"Noisy Batch Shape: {len(noisy_batch)}")
#         print(f"Noisy Batch member type: {type(noisy_batch[0])}")
#         print(f"Noisy Batch member shape: {noisy_batch[0].shape}")
#     break


# train_loader:  22
# GT Batch type: <class 'list'>
# GT Batch Shape: 10
# GT Batch member type: <class 'torch.Tensor'>
# GT Batch member shape: torch.Size([32, 3, 128, 128])
# Noisy Batch type: <class 'list'>
# Noisy Batch Shape: 10
# Noisy Batch member type: <class 'torch.Tensor'>
# Noisy Batch member shape: torch.Size([32, 3, 128, 128])


# test_loader:  6
# GT Batch type: <class 'list'>
# GT Batch Shape: 10
# GT Batch member type: <class 'torch.Tensor'>
# GT Batch member shape: torch.Size([32, 3, 128, 128])
# Noisy Batch type: <class 'list'>
# Noisy Batch Shape: 10
# Noisy Batch member type: <class 'torch.Tensor'>
# Noisy Batch member shape: torch.Size([32, 3, 128, 128])

