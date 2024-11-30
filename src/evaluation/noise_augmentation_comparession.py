
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

def plot_noise_sampling_results(results):
    

    noise_types = list(results.keys())
    sampling_methods = list(results[noise_types[0]].keys())

    # Plot Final PSNR Improvement for each Noise Type and Sampling Method
    plt.figure(figsize=(15, 10))

    for i, noise_type in enumerate(noise_types):
        final_psnr = [results[noise_type][method]['final_psnr_improvement'] for method in sampling_methods]
        final_mse = [results[noise_type][method]['final_mse'] for method in sampling_methods]

        plt.subplot(len(noise_types), 2, 2 * i + 1)
        plt.bar(sampling_methods, final_psnr, color='skyblue')
        plt.title(f'PSNR Improvement for {noise_type} Noise (Level {noise_level}%)')
        plt.xlabel('Sampling Method')
        plt.ylabel('Final PSNR Improvement (dB)')
        plt.xticks(rotation=45)

        plt.subplot(len(noise_types), 2, 2 * i + 2)
        plt.bar(sampling_methods, final_mse, color='lightgreen')
        plt.title(f'MSE for {noise_type} Noise (Level {noise_level}%)')
        plt.xlabel('Sampling Method')
        plt.ylabel('Final MSE')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('noise_sampling_comparison.png')
    plt.show()


def noise_augmentation_sampling_experiment(
    df_sample,
    model,
    initial_train_indices,
    test_indices,
    available_pool_indices,
    random_seed,
    max_pixel,
    iterations,
    sampling_methods_args,
    noise_types_args,  
    noise_level=25  # Fixed noise level
):
    """
    Experiment with different noise types and sampling methods during training.

    Args:
    - df_sample: DataFrame containing the dataset.
    - model: The denoising model to be used.
    - initial_train_indices: Indices of the initial training set.
    - test_indices: Indices of the test set.
    - available_pool_indices: Indices of the pool for active learning.
    - random_seed: Random seed for reproducibility.
    - max_pixel: Maximum pixel value of the dataset (for PSNR calculation).
    - iterations: Number of active learning iterations.
    - sampling_methods_args: List of sampling strategies to compare.
    - noise_types_args: List of noise types to apply during training.
    - noise_level: Noise level percentage to apply during training.

    Returns:
    - results: Nested dictionary containing MSE and PSNR improvements for each noise type and sampling method.
    """

    results = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    # model = LightDenoisingUNet().to(device)
    # model = LightRIDNet().to(device)

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

    # print(noise_types_args)
    # print(sampling_methods_args)
    for noise in noise_types_args:
        # print(noise)
        results[noise] = {}

        for method in sampling_methods_args:
            print(f"\n--- Experimenting with Noise Type: {noise}, Sampling Method: {method} ---")

            # Create a noisy dataset with the specified noise type and level
            noisy_dataset = DenoisingPatchDataset(df_sample,
                                          normalize=normalize_transform,
                                          transform=None,
                                          active_indices=None,
                                          addNoise=True,
                                          noise_type=noise,
                                          noise_level=noise_level
                                                  )
            if sample_dataset.normalize:
                max_pixel = 1.0
            else:
                max_pixel = 255.0
            # Reset the model for each experiment
            model_instance = LightRIDNet().to(device)

            # Initialize Active Learning Pipeline
            al_pipeline = ActiveLearningPipeline(
                model=model_instance,
                train_indices=initial_train_indices.copy(),
                test_indices=test_indices,
                available_pool_indices=available_pool_indices.copy(),
                selection_criterion=method,
                iterations=iterations,
                budget_per_iter=128,  # Fixed budget for simplicity
                dataset=noisy_dataset,
                max_pixel=max_pixel,
                random_seed=random_seed,
                batch_size=64,
                train_all=True
            )

            # Run pipeline and store results
            mse_scores, psnr_improvement_scores = al_pipeline.run_pipeline()

            # Store results
            results[noise][method] = {
                'mse_scores': mse_scores,
                'psnr_improvement_scores': psnr_improvement_scores,
                'final_mse': mse_scores[-1],
                'final_psnr_improvement': psnr_improvement_scores[-1]
            }

    return results




def plot_results_by_noise_type(results):
    """
    Plot the adjusted PSNR improvement and MSE over iterations separately for each noise type.

    Args:
    - results: Dictionary containing adjusted PSNR improvement and MSE scores.
    """
    noise_types = list(results.keys())
    sampling_methods = list(results[noise_types[0]].keys())

    # Iterate through each noise type
    for noise_type in noise_types:
        plt.figure(figsize=(15, 6))

        # Plot PSNR Improvement over iterations for the current noise type
        plt.subplot(1, 2, 1)
        for method in sampling_methods:
            psnr_scores = results[noise_type][method]['psnr_improvement_scores']
            plt.plot(
                range(1, len(psnr_scores) + 1),
                psnr_scores,
                label=method
            )
        plt.title(f'Adjusted PSNR Improvement over Iterations ({noise_type} Noise)')
        plt.xlabel('Iteration')
        plt.ylabel('Adjusted PSNR Improvement (dB)')
        plt.legend(loc='lower right')
        plt.grid(True)

        # Plot MSE over iterations for the current noise type
        plt.subplot(1, 2, 2)
        for method in sampling_methods:
            mse_scores = results[noise_type][method]['mse_scores']
            plt.plot(
                range(1, len(mse_scores) + 1),
                mse_scores,
                label=method
            )
        plt.title(f'MSE over Iterations ({noise_type} Noise)')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{noise_type}_noise_performance.png')
        plt.show()


# Define sampling methods and experiment parameters
sampling_methods=['random', 'uncertainty', 'noise_pattern_diversity', 'noise_pattern_diversity_2']
noise_types=['gaussian', 'salt_pepper', 'speckle']  # Noise types to test
noise_level=25  # Fixed noise level

# Run the noise augmentation experiment with sampling methods
noise_sampling_results = noise_augmentation_sampling_experiment(
    df_sample=df_sample,
    model=LightRIDNet(),
    initial_train_indices=initial_train_indices,
    test_indices=test_indices,
    available_pool_indices=available_pool_indices,
    random_seed=random_seed,
    max_pixel=max_pixel,
    iterations=10,
    sampling_methods_args=sampling_methods,
    noise_types_args=noise_types,
    noise_level=noise_level
)

# Visualize the results
plot_noise_sampling_results(noise_sampling_results)


plot_results_by_noise_type(noise_sampling_results)

def adjust_psnr_results(results, baseline_psnr=28.57, noisy_psnr_values=None):
    """
    Adjust the PSNR improvement values by subtracting the difference
    between the baseline PSNR and the noisy PSNR for each noise type.

    Args:
    - results: Original results dictionary with PSNR improvement scores.
    - baseline_psnr: PSNR of the clean images (default is 28.57).
    - noisy_psnr_values: Dictionary with noisy PSNR values for each noise type.

    Returns:
    - adjusted_results: Dictionary with adjusted PSNR improvements.
    """

    if noisy_psnr_values is None:
        noisy_psnr_values = {
            'gaussian': 16.39,
            'salt_pepper': 26.91,
            'speckle': 21.34,  
        }

    adjusted_results = {}

    for noise_type, methods in results.items():
        adjusted_results[noise_type] = {}

        for method, metrics in methods.items():
            final_psnr_improvement = metrics['final_psnr_improvement']

            
            
            noisy_psnr = noisy_psnr_values[noise_type]

            # Adjust the PSNR improvement
            psnr_adjustment = baseline_psnr - noisy_psnr
            adjusted_psnr_improvement = final_psnr_improvement - psnr_adjustment

            # Store the adjusted results
            adjusted_results[noise_type][method] = {
                'mse_scores': metrics['mse_scores'],
                'psnr_improvement_scores': [
                    score - psnr_adjustment for score in metrics['psnr_improvement_scores']
                ],
                'final_mse': metrics['final_mse'],
                'final_psnr_improvement': adjusted_psnr_improvement,
            }

    return adjusted_results

# Adjust the PSNR results
adjusted_results = adjust_psnr_results(noise_sampling_results)

# Example print to verify
for noise_type, methods in adjusted_results.items():
    for method, metrics in methods.items():
        print(f"Noise Type: {noise_type}, Method: {method}, Adjusted Final PSNR Improvement: {metrics['final_psnr_improvement']:.2f} dB")


plot_noise_sampling_results(adjusted_results)

plot_results_by_noise_type(adjusted_results)