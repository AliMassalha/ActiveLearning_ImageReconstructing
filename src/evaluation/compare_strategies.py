
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


def results_comparing(
    model,
    initial_train_indices,
    test_indices,
    available_pool_indices,
    iterations,
    budget_per_iter,
    dataset,
    batch_size,
    max_pixel,
    sampling_methods,
    random_seed,
    train_all=True):
    """
    Run comprehensive active learning experiments across different sampling strategies

    Args:
    - model: Your image denoising neural network model
    - dataset: The complete dataset
    - sampling_methods: List of sampling methods to compare
    """
    # Experiment results storage
    results = {}



    # Reproducibility
    # np.random.seed(42)
    # torch.manual_seed(42)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Experiment for each sampling method
    for sampling_method in sampling_methods:
        # Reset model parameters for each experiment
        # model = LightDenoisingUNet().to(device)
        # model = Generator1(64,64,3).to(device)
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
            iterations=iterations,
            budget_per_iter=budget_per_iter,
            batch_size=batch_size,
            dataset=dataset,
            max_pixel=max_pixel,
            random_seed=random_seed,
            train_all=train_all
        )

        # Run the pipeline
        mse_scores, psnr_improvement_scores = al_pipeline.run_pipeline()

        # Store results
        results[sampling_method] = {
            'mse_scores': mse_scores,
            'psnr_improvement_scores': psnr_improvement_scores
        }
        print(f"Results for {sampling_method}:\n MSE Scores: {mse_scores}\n PSNR Improvement Scores: {psnr_improvement_scores}")



    return results


# Example usage
sampling_methods = [
    'noise_pattern_diversity',
    'noise_pattern_diversity_2',
    'random',
    'uncertainty',
    'diversity',
    # 'diversity_2',
    'entropy',
    # 'margin',
    'gradient',

]


results = results_comparing(
    model=model,
    initial_train_indices=initial_train_indices,
    test_indices=test_indices,
    available_pool_indices=available_pool_indices,
    iterations=15,
    budget_per_iter=128,
    dataset=sample_dataset,
    batch_size=batch_size,
    max_pixel=max_pixel,
    sampling_methods=sampling_methods,
    random_seed=random_seed,
    train_all=True
)


# Visualization functions
def plot_learning_curves(results):
    plt.figure(figsize=(15, 5))

    # PSNR Improvement Plot
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        plt.plot(data['psnr_improvement_scores'], label=method)
    plt.title('PSNR Improvement Over Active Learning Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR Improvement (dB)')
    plt.legend()

    # MSE Plot
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        plt.plot(data['mse_scores'], label=method)
    plt.title('MSE Over Active Learning Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig('active_learning_curves.png')
    plt.close()

def plot_final_performance_comparison(results):
    methods = list(results.keys())
    final_psnr_improvements = [
        results[method]['psnr_improvement_scores'][-1] for method in methods
    ]
    final_mse_scores = [
        results[method]['mse_scores'][-1] for method in methods
    ]

    plt.figure(figsize=(12, 5))

    # PSNR Improvement Bar Chart
    plt.subplot(1, 2, 1)
    plt.bar(methods, final_psnr_improvements)
    plt.title('Final PSNR Improvement by Sampling Method')
    plt.xlabel('Sampling Method')
    plt.ylabel('PSNR Improvement (dB)')
    plt.xticks(rotation=45)

    # MSE Bar Chart
    plt.subplot(1, 2, 2)
    plt.bar(methods, final_mse_scores)
    plt.title('Final MSE by Sampling Method')
    plt.xlabel('Sampling Method')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('final_performance_comparison.png')
    plt.close()

# Generate visualizations
plot_learning_curves(results)
plot_final_performance_comparison(results)





def advanced_active_learning_analysis(results, output_dir='al_experiments'):
    """
    Perform advanced analysis on active learning experiment results

    Args:
    - results: Dictionary of experiment results
    - output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Statistical Significance Analysis
    def perform_statistical_tests(results):
        methods = list(results.keys())
        psnr_comparisons = {}
        mse_comparisons = {}

        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                # PSNR improvement t-test
                t_stat, p_value = stats.ttest_ind(
                    results[method1]['psnr_improvement_scores'],
                    results[method2]['psnr_improvement_scores']
                )
                psnr_comparisons[f"{method1}_vs_{method2}"] = { # Changed this line
                    't_statistic': t_stat,
                    'p_value': p_value
                }

                # MSE t-test
                t_stat, p_value = stats.ttest_ind(
                    results[method1]['mse_scores'],
                    results[method2]['mse_scores']
                )
                mse_comparisons[f"{method1}_vs_{method2}"] = { # Changed this line
                    't_statistic': t_stat,
                    'p_value': p_value
                }

        # Save statistical tests results
        with open(os.path.join(output_dir, 'statistical_tests.json'), 'w') as f:
            json.dump({
                'psnr_comparisons': psnr_comparisons,
                'mse_comparisons': mse_comparisons
            }, f, indent=2)

        return psnr_comparisons, mse_comparisons

    # 2. Convergence Analysis
    def analyze_convergence(results):
        convergence_metrics = {}

        for method, data in results.items():
            psnr_improvements = data['psnr_improvement_scores']
            mse_scores = data['mse_scores']

            # Calculate rate of improvement
            psnr_improvement_rate = np.diff(psnr_improvements)
            mse_improvement_rate = -np.diff(mse_scores)  # Negative because we want to see reduction

            convergence_metrics[method] = {
                'final_psnr_improvement': psnr_improvements[-1],
                'final_mse': mse_scores[-1],
                'psnr_improvement_rate': psnr_improvement_rate.tolist(),
                'mse_improvement_rate': mse_improvement_rate.tolist(),
                'psnr_stabilization_iteration': int(np.argmin(np.abs(psnr_improvement_rate))),
                'mse_stabilization_iteration': int(np.argmin(np.abs(mse_improvement_rate)))
            }

        # Save convergence analysis
        with open(os.path.join(output_dir, 'convergence_analysis.json'), 'w') as f:
            json.dump(convergence_metrics, f, indent=2)

        return convergence_metrics

    # 3. Detailed Visualization
    def create_advanced_visualizations(results):
        # Boxplot of final performances
        plt.figure(figsize=(15, 5))

        # PSNR Improvement Boxplot
        plt.subplot(1, 2, 1)
        psnr_data = [results[method]['psnr_improvement_scores'] for method in results]
        plt.boxplot(psnr_data, labels=list(results.keys()))
        plt.title('PSNR Improvement Distribution')
        plt.ylabel('PSNR Improvement (dB)')
        plt.xticks(rotation=45)

        # MSE Boxplot
        plt.subplot(1, 2, 2)
        mse_data = [results[method]['mse_scores'] for method in results]
        plt.boxplot(mse_data, labels=list(results.keys()))
        plt.title('MSE Distribution')
        plt.ylabel('Mean Squared Error')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_distribution.png'))
        plt.close()

        # Convergence Rate Visualization
        plt.figure(figsize=(15, 5))

        # PSNR Improvement Rate
        plt.subplot(1, 2, 1)
        for method, data in results.items():
            psnr_improvement_rate = np.diff(data['psnr_improvement_scores'])
            plt.plot(psnr_improvement_rate, label=method)
        plt.title('PSNR Improvement Rate')
        plt.xlabel('Iteration')
        plt.ylabel('PSNR Improvement Rate')
        plt.legend()

        # MSE Improvement Rate
        plt.subplot(1, 2, 2)
        for method, data in results.items():
            mse_improvement_rate = -np.diff(data['mse_scores'])
            plt.plot(mse_improvement_rate, label=method)
        plt.title('MSE Reduction Rate')
        plt.xlabel('Iteration')
        plt.ylabel('MSE Reduction Rate')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_rates.png'))
        plt.close()

    # 4. Comprehensive Results Saving
    def save_comprehensive_results(results):
        # Save raw results
        with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Create a DataFrame for easier analysis
        results_df = pd.DataFrame({
            method: {
                'final_psnr_improvement': data['psnr_improvement_scores'][-1],
                'final_mse': data['mse_scores'][-1]
            } for method, data in results.items()
        }).T

        results_df.to_csv(os.path.join(output_dir, 'results_summary.csv'))

    # Run all analyses
    statistical_tests = perform_statistical_tests(results)
    convergence_analysis = analyze_convergence(results)
    create_advanced_visualizations(results)
    save_comprehensive_results(results)

    return {
        'statistical_tests': statistical_tests,
        'convergence_analysis': convergence_analysis
    }

# Modify your existing experiment run to include this
additional_analysis = advanced_active_learning_analysis(results)