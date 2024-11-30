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

def analyze_labeling_budget_performance(
    model,
    df,
    iterations,
    budget_per_iter,
    batch_size,
    max_pixel,
    sampling_methods,
    random_seed,
    train_all=True,
    budget_percentages=[0.1, 0.15]
    ):
    """
    Analyze model performance across different labeling budgets

    Args:
    - model: Your image denoising neural network model
    - dataset: The complete dataset
    - sampling_methods: List of sampling methods to compare
    - budget_percentages: Percentage of dataset to use for labeling

    Returns:
    - Dictionary of performance results for each method and budget
    """
    results = {}



    # Iterate through each labeling budget
    for budget in budget_percentages:
        results[budget] = {}
        df_sample = df.sample(frac=budget,random_state=42).reset_index(drop=True)
        initial_train_fraction = 0.1
        test_fraction = 0.2
        train_pool_df, test_df = train_test_split(df_sample, test_size=test_fraction, random_state=42)
        initial_train_df, pool_df = train_test_split(train_pool_df, test_size=1 - initial_train_fraction, random_state=42)

        initial_train_indices = initial_train_df.index.tolist()
        test_indices = test_df.index.tolist()
        available_pool_indices = pool_df.index.tolist()
        print(f"Initial Train Size: {len(initial_train_indices)}")
        print(f"Test Size: {len(test_indices)}")
        print(f"Available Pool Size: {len(available_pool_indices)}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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



        sample_dataset = DenoisingPatchDataset(df_sample,
                                              normalize=normalize_transform,
                                              transform=None,
                                              active_indices=None)



        # Experiment for each sampling method
        for sampling_method in sampling_methods:
            # Reset model parameters
            model = LightRIDNet().to(device)

            # Create a copy of initial states for fair comparison
            train_indices = initial_train_indices.copy()
            pool_indices = available_pool_indices.copy()

            # Initialize Active Learning Pipeline
            al_pipeline = ActiveLearningPipeline(
                model=model,
                train_indices=train_indices,
                test_indices=test_indices,
                available_pool_indices=pool_indices,
                selection_criterion=sampling_method,
                iterations=iterations,  # Reduced iterations for budget analysis
                budget_per_iter=budget_per_iter,  # Adjust based on budget size
                dataset=sample_dataset,
                max_pixel=max_pixel  # Adjust based on your data normalization
            )

            # Run the pipeline
            mse_scores, psnr_improvement_scores = al_pipeline.run_pipeline()

            # Store results
            results[budget][sampling_method] = {
                'mse_scores': mse_scores,
                'psnr_improvement_scores': psnr_improvement_scores,
                'final_mse': mse_scores[-1],
                'final_psnr_improvement': psnr_improvement_scores[-1]
            }

    # Visualization
    def plot_budget_performance(results):
        # Prepare data for plotting
        budget_levels = list(results.keys())
        sampling_methods = list(list(results.values())[0].keys())

        # PSNR Improvement Plot
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        for method in sampling_methods:
            psnr_improvements = [
                results[budget][method]['final_psnr_improvement'] for budget in budget_levels
            ]
            plt.plot(budget_levels, psnr_improvements, marker='o', label=method)
        plt.title('PSNR Improvement vs Labeling Budget')
        plt.xlabel('Labeling Budget (% of Dataset)')
        plt.ylabel('PSNR Improvement (dB)')
        plt.legend()

        # MSE Plot
        plt.subplot(1, 2, 2)
        for method in sampling_methods:
            mse_scores = [
                results[budget][method]['final_mse'] for budget in budget_levels
            ]
            plt.plot(budget_levels, mse_scores, marker='o', label=method)
        plt.title('MSE vs Labeling Budget')
        plt.xlabel('Labeling Budget (% of Dataset)')
        plt.ylabel('Mean Squared Error')
        plt.legend()

        plt.tight_layout()
        plt.savefig('labeling_budget_performance.png')
        plt.close()

    # Generate visualization
    plot_budget_performance(results)

    # Save results to JSON for detailed analysis
    with open('labeling_budget_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


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

budget_performance = analyze_labeling_budget_performance(
    model=model,
    df=df,
    iterations=15,
    budget_per_iter=128,
    batch_size=batch_size,
    max_pixel=max_pixel,
    sampling_methods=sampling_methods,
    random_seed=random_seed,
    train_all=True
)