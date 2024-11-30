import matplotlib.pyplot as plt
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
# print(f"Max pixel value: {max_pixel}")


def plot_ablation_results(ablation_results):
    plt.figure(figsize=(12, 6))

    # PSNR Improvement Plot
    for feature, result in ablation_results.items():
        plt.plot(result['psnr_improvement_scores'], label=f"Without {feature}")

    plt.title('PSNR Improvement Across Ablations')
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('PSNR Improvement (dB)')
    plt.legend()
    plt.savefig('ablation_psnr_improvement.png')
    plt.show()

    # MSE Plot
    plt.figure(figsize=(12, 6))
    for feature, result in ablation_results.items():
        plt.plot(result['mse_scores'], label=f"Without {feature}")

    plt.title('MSE Across Ablations')
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('ablation_mse.png')
    plt.show()


def run_ablation_study(pipeline_class, model, initial_train_indices, test_indices,
                       available_pool_indices, dataset,random_seed,batch_size,max_pixel,
                       selection_criterion='noise_pattern_diversity_ablation'):
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for feature in ablation_features:
        model = LightRIDNet().to(device)

        # Create a copy of initial states for fair comparison
        train_indices = initial_train_indices.copy()
        pool_indices = available_pool_indices.copy()

        print(f"\nRunning ablation study: Removing {feature}...")
        pipeline = pipeline_class(
            model=model,
            train_indices=train_indices,
            test_indices=test_indices,
            available_pool_indices=available_pool_indices,
            selection_criterion=selection_criterion,
            iterations=10,
            budget_per_iter=128,
            dataset=dataset,
            random_seed=random_seed,
            batch_size=batch_size,
            max_pixel=max_pixel,
            train_all=True
        )

        # pipeline._noise_pattern_diversity_sampling = lambda: pipeline._noise_pattern_diversity_sampling_ablation(
        #     ablation_components=[feature]
        # )

        mse_scores, psnr_improvement_scores = pipeline.run_pipeline()

        results[feature] = {
            'mse_scores': mse_scores,
            'psnr_improvement_scores': psnr_improvement_scores,
        }

    return results

import json
import os

def save_results_as_json(results, filename='ablation_results.json', output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    filepath = os.path.join(output_dir, filename)

    # Convert tensors to lists for JSON serialization
    json_results = {
        feature: {
            'mse_scores': [float(score) for score in data['mse_scores']],
            'psnr_improvement_scores': [float(score) for score in data['psnr_improvement_scores']]
        }
        for feature, data in results.items()
    }

    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved as JSON: {filepath}")

ablation_features = [
    'reconstruction_error',
    'model_uncertainty',
    'prediction_variance',
    'dx',
    'dy',
    'quadrant_variance',
    'high_freq_content',
    'entropy',
    'spatial_correlation',
]




# Run and collect results
ablation_results = run_ablation_study(ActiveLearningPipeline, model, initial_train_indices,
                                      test_indices, available_pool_indices, sample_dataset,random_seed,
                                      batch_size,max_pixel)

plot_ablation_results(ablation_results)

# Save the ablation results as JSON
save_results_as_json(ablation_results)









ablation_features = [
    'reconstruction_error',
    'prediction_variance',
    'dx',
    'dy',
    'noise_prediction_difference',
    'spectral_energy'
    
]

# Run and collect results
ablation_results = run_ablation_study(ActiveLearningPipeline, model, initial_train_indices,
                                      test_indices, available_pool_indices, sample_dataset,random_seed,
                                      batch_size,max_pixel,'noise_pattern_diversity_ablation_2')


plot_ablation_results(ablation_results)

# Save the ablation results as JSON
save_results_as_json(ablation_results)


