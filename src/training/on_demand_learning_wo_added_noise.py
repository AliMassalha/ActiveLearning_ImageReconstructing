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
        self.batchSize = 100
        self.fine_size = 256  # Updated for your 256x256 patches
        self.nef = 64  # encoder filters
        self.ngf = 64  # generator filters
        self.nc = 3    # channels
        self.lr = 0.001
        self.beta1 = 0.5
        self.nThreads = 0 ##########change to zero
        self.manualSeed = 1234
        self.n_epochs = 15
        self.gpu = True if torch.cuda.is_available() else False
        self.name = 'denoise'
        # self.subtasks = [20, 40, 60, 80, 100]  # Noise levels
        # self.batchSize_per_subtask = [self.batchSize // len(self.subtasks)] * len(self.subtasks)
        # On-demand learning parameters
        self.N = 3  # number of noise level groups
        self.B = self.batchSize  # total batch size
        self.display_iter = 50

opt = Options()

class OnDemandDenoiser:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt.gpu else "cpu")
        self.random_seed = opt.manualSeed
        

        # self.model = Generator1(opt.nef,opt.ngf,opt.nc).to(self.device)
        # self.model = LightDenoisingUNet().to(self.device)
        self.model = LightRIDNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt.lr,
            # betas=(opt.beta1, 0.999)
        )

        # Initialize batch sizes for each noise group
        self.Bi = torch.ones(opt.N) * (opt.B / opt.N)
        self._set_global_seeds()

    def _set_global_seeds(self):
        """Set seeds for all random number generators."""
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    def validate_group(self, dataset, group_idx, max_pixel):
        """Validate model on a specific noise level group"""
        self.model.eval()
        group_indices = dataset.get_group_indices(group_idx)
        group_dataset = Subset(dataset, group_indices)
        group_loader = DataLoader(group_dataset, batch_size=32, shuffle=False)

        total_psnr = 0
        count = 0

        with torch.no_grad():
            for noisy, clean in group_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                restored = self.model(noisy)

                # Compute PSNR
                mse = torch.mean((restored - clean) ** 2)
                psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
                total_psnr += psnr.item()
                count += 1

        return total_psnr / count if count > 0 else 0

    def update_batch_distribution(self, psnr_values):
        """Update batch size distribution based on PSNR values"""
        psnr_sum = sum(1/p for p in psnr_values)
        for i in range(self.opt.N):
            self.Bi[i] = (1/psnr_values[i] / psnr_sum) * self.opt.B

    def train(self, train_dataset, test_dataset ,max_pixel=1.0):
        for epoch in range(self.opt.n_epochs):
            self.model.train()
            running_loss = 0.0

            # Create batch indices for each noise group based on current distribution
            group_batches = []
            for i in range(self.opt.N):
                group_indices = train_dataset.get_group_indices(i)
                # print(f"Group {i}: {len(group_indices)} samples")
                # print(group_indices)
                # print(f"Group {i} batch size: {int(self.Bi[i])}")
                n_samples = int(self.Bi[i])
                if n_samples > 0:
                    selected_indices = np.random.choice(
                        group_indices,
                        size=min(n_samples, len(group_indices)),
                        replace=True
                    )
                    group_batches.extend(selected_indices)

            # Create dataloader with current distribution
            train_subset = Subset(train_dataset, group_batches)
            train_loader = DataLoader(
                train_subset,
                batch_size=self.opt.batchSize,
                shuffle=True,
                num_workers=0
            )

            # Training loop
            for i, (noisy, clean) in tqdm(enumerate(train_loader)):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                # Forward pass
                restored = self.model(noisy)
                loss = self.criterion(restored, clean)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Validate and update batch distribution
            psnr_values = []
            for group_idx in range(self.opt.N):
                psnr = self.validate_group(test_dataset, group_idx, max_pixel)
                psnr_values.append(psnr)

            self.update_batch_distribution(psnr_values)

            print(f"Epoch {epoch+1}/{self.opt.n_epochs}")
            print(f"Average loss: {running_loss/len(train_loader):.4f}")
            print(f"PSNR values: {psnr_values}")
            print(f"New batch distribution: {self.Bi.tolist()}")
            os.makedirs('checkpoints', exist_ok=True)
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'batch_distribution': self.Bi,
                }, f'checkpoints/{self.opt.name}_epoch_{epoch+1}.pth')



# Load your dataset
patches_dir = 'data/patches'
df = get_patch_pair_paths(patches_dir)
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

train_dataset = NoiseGroupedDenoisingDataset(train_df,group=True,normalize=normalize_transform)
test_dataset = NoiseGroupedDenoisingDataset(test_df,group=True,normalize=normalize_transform)

if train_dataset.normalize:
  max_pixel = 1.0
else:
  max_pixel = 255.0

# LightRIDNet

# Initialize and train the denoiser
denoiser = OnDemandDenoiser(opt)
denoiser.train(train_dataset, test_dataset, max_pixel)



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


device = torch.device("cuda" if opt.gpu else "cpu")
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
evaluate_psnr(denoiser.model, test_loader, device, max_pixel)