import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from scipy import stats

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

# from models.unet import LightDownBlock, LightUpBlock, LightDenoisingUNet
from models.ridnet import LightEAM, LightRIDNet
from training.train_active_learning import ActiveLearningPipeline


patches_dir = 'data/patches'
df = get_patch_pair_paths(patches_dir)

df_sample = df.sample(frac=0.2,random_state=42).reset_index(drop=True)

# Fraction of data to initially label and use for training
initial_train_fraction = 0.1  # Start with 10% of data as labeled
test_fraction = 0.2           # 20% of the total data for testing

# Split data into train+pool and test sets
train_pool_df, test_df = train_test_split(df_sample, test_size=test_fraction, random_state=42)

# Further split train+pool into initial train and unlabeled pool
initial_train_df, pool_df = train_test_split(train_pool_df, test_size=1 - initial_train_fraction, random_state=42)

# Convert these DataFrames into index lists for the pipeline
initial_train_indices = initial_train_df.index.tolist()
test_indices = test_df.index.tolist()
available_pool_indices = pool_df.index.tolist()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LightDenoisingUNet().to(device)
model = LightRIDNet().to(device)

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
random_seed = 1234

sample_dataset = DenoisingPatchDataset(df_sample,
                                       normalize=normalize_transform,
                                       transform=None,
                                       active_indices=None)

if sample_dataset.normalize:
    max_pixel = 1.0
else:
    max_pixel = 255.0


def budget_per_iteration_experiment(
    df_sample,
    model,
    initial_train_indices, test_indices, available_pool_indices,
    random_seed,
    max_pixel,
    iterations,
    batch_size,
    sampling_methods):  # Accept list of sampling methods

    # Define different budget sizes to test
    budget_sizes = [32, 64, 256]
    results = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for budget in budget_sizes:
        results[budget] = {}  # Initialize results for each budget size

        for sampling_method in sampling_methods:
            print(f"\n--- Experimenting with Budget: {budget}, Sampling Method: {sampling_method} ---")

            # Reset the model for each experiment
            model_instance = LightRIDNet().to(device)

            # Create a copy of initial states for fair comparison
            train_indices = initial_train_indices.copy()
            pool_indices = available_pool_indices.copy()

            # Initialize Active Learning Pipeline
            al_pipeline = ActiveLearningPipeline(
                model=model_instance,
                train_indices=train_indices,
                test_indices=test_indices,
                available_pool_indices=pool_indices,
                selection_criterion=sampling_method,
                iterations=iterations,
                budget_per_iter=budget,
                dataset=sample_dataset,
                max_pixel=max_pixel,
                random_seed=random_seed,
                batch_size=batch_size,
                train_all=True
            )

            # Run pipeline and store results
            mse_scores, psnr_improvement_scores = al_pipeline.run_pipeline()

            results[budget][sampling_method] = {
                'mse_scores': mse_scores,
                'psnr_improvement_scores': psnr_improvement_scores,
                'final_mse': mse_scores[-1],
                'final_psnr_improvement': psnr_improvement_scores[-1]
            }

    return results


def plot_budget_experiment_results(results):
    import matplotlib.pyplot as plt

    budgets = list(results.keys())

    for budget in budgets:
        sampling_methods = list(results[budget].keys())

        # Extract final results for each sampling method
        final_mse = [results[budget][method]['final_mse'] for method in sampling_methods]
        final_psnr_improvement = [results[budget][method]['final_psnr_improvement'] for method in sampling_methods]

        # Create plots for MSE and PSNR Improvement
        plt.figure(figsize=(12, 5))

        # MSE Plot
        plt.subplot(1, 2, 1)
        plt.bar(sampling_methods, final_mse, color='skyblue')
        plt.title(f'Final MSE for Budget {budget}')
        plt.xlabel('Sampling Method')
        plt.ylabel('Final MSE')
        plt.xticks(rotation=45)

        # PSNR Improvement Plot
        plt.subplot(1, 2, 2)
        plt.bar(sampling_methods, final_psnr_improvement, color='lightgreen')
        plt.title(f'Final PSNR Improvement for Budget {budget}')
        plt.xlabel('Sampling Method')
        plt.ylabel('Final PSNR Improvement (dB)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f'budget_{budget}_comparison.png')  # Save each plot
        plt.show()


sampling_methods = [
    'noise_pattern_diversity',
    'noise_pattern_diversity_2',
    'random',
    # 'uncertainty',
    # 'diversity',
    # 'diversity_2',
    # 'entropy',
    # 'margin',
    # 'gradient',

]



# Run experiments
results = budget_per_iteration_experiment(
    df_sample=df_sample,
    model=LightRIDNet(),
    initial_train_indices=initial_train_indices,
    test_indices=test_indices,
    available_pool_indices=available_pool_indices,
    random_seed=random_seed,
    batch_size=batch_size,
    max_pixel=max_pixel,
    iterations=10,
    sampling_methods=sampling_methods
)

# Plot the results
plot_budget_experiment_results(results)
