import time
from scipy.stats import entropy
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_squared_error


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

class ActiveLearningPipeline:
    def __init__(self, model, train_indices, test_indices, available_pool_indices,
                 selection_criterion, iterations, budget_per_iter, dataset,
                 max_pixel, batch_size=64,random_seed=None,train_all=True):

        self.random_seed = random_seed
        if self.random_seed is not None:
          self._set_global_seeds()
          # Ensure model is deterministic
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = False

        self.train_all = train_all

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.dataset = dataset
        self.batch_size = batch_size

        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.selection_criterion = selection_criterion

        self.train_indices = train_indices
        self.test_indices = test_indices
        self.available_pool_indices = available_pool_indices
        self.newly_added_indices = []

        self.max_pixel = max_pixel


        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Create data loader for test data
        self.test_subset = torch.utils.data.Subset(self.dataset, self.test_indices)
        self.test_loader = DataLoader(self.test_subset, batch_size=self.batch_size, shuffle=False,
                                      # worker_init_fn=self._worker_init_fn
                                      )

    def _set_global_seeds(self):
        """Set seeds for all random number generators."""
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Ensure full determinism for CUDA operations
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)

    def _worker_init_fn(self, worker_id):
        """Ensure reproducibility for DataLoader workers."""
        worker_seed = self.random_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def run_pipeline(self):
        """
        Run the active learning pipeline
        :return: list of average MSE losses at each iteration
        """
        mse_scores = []
        psnr_improvment_scores = []
        # Reset seeds before each pipeline run to ensure absolute reproducibility
        # self._set_global_seeds()

        for iteration in range(self.iterations):
            print(f"\n--- Active Learning Iteration {iteration + 1} ---")


            # print("Training model...")

            # Ensure reproducible training
            # self._set_global_seeds()
            if self.train_all:
                # Train model on current training set
                self._train_model()
            else:
                if iteration == 0:
                    # First iteration, train on initial training set
                    self._train_model()
                else:
                    # Subsequent iterations, train only on newly added data
                    self._train_model(train_indices=self.newly_added_indices)

            # Evaluate model on the test set
            print("Evaluating model...")
            # Ensure reproducible evaluating
            # self._set_global_seeds()
            mse, mean_psnr_noisy, mean_psnr_denoised = self._evaluate_model()
            mse_scores.append(mse)
            psnr_improvment_scores.append(mean_psnr_denoised - mean_psnr_noisy)
            print(f"Iteration [{iteration+1}/{self.iterations}], Test MSE: {mse:.4f}")
            print(f"\nResults:")
            print(f"Mean PSNR of noisy images: {mean_psnr_noisy:.2f} dB")
            print(f"Mean PSNR of denoised images: {mean_psnr_denoised:.2f} dB")
            print(f"PSNR improvement: {mean_psnr_denoised - mean_psnr_noisy:.2f} dB")

            # Select new samples from the pool and add to training set
            if iteration < self.iterations - 1:
              # self._set_global_seeds()
              self._select_and_update()

        return mse_scores, psnr_improvment_scores

    def _train_model(self, train_indices=None):
        # if not self.train_indices:
        #   raise ValueError("Training indices are empty. Cannot proceed with training.")
        if train_indices is None:
            train_indices = self.train_indices
        # Create data loader for training data
        # print("Creating data loader for training data...")
        # print(self.train_indices)
        train_subset = torch.utils.data.Subset(self.dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                  # worker_init_fn=self._worker_init_fn
                                  )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        print("Training model...")
        for noisy_patches, clean_patches in tqdm(train_loader):
            # print("Training model...(inside loop)")
            noisy_patches, clean_patches = noisy_patches.to(self.device), clean_patches.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(noisy_patches).to(self.device)
            loss = self.criterion(outputs, clean_patches)
            loss.backward()
            optimizer.step()

    def _evaluate_model(self):
        def calculate_psnr(img1, img2, max_val):
            # Ensure the images are in range [0, 1]
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = max_val
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            return psnr.item()



        self.model.eval()
        total_loss = 0.0
        total_psnr_noisy = 0
        total_psnr_denoised = 0
        num_batches = len(self.test_loader)

        with torch.no_grad():
            for noisy_patches, clean_patches in tqdm(self.test_loader):
                noisy_patches, clean_patches = noisy_patches.to(self.device), clean_patches.to(self.device)
                outputs = self.model(noisy_patches).to(self.device)
                # Calculate PSNR for each image in batch
                for i in range(clean_patches.size(0)):
                    # Original noisy image PSNR
                    psnr_noisy = calculate_psnr(clean_patches[i], noisy_patches[i],self.max_pixel)
                    total_psnr_noisy += psnr_noisy

                    # Denoised image PSNR
                    psnr_denoised = calculate_psnr(clean_patches[i], outputs[i],self.max_pixel)
                    total_psnr_denoised += psnr_denoised
                loss = self.criterion(outputs, clean_patches)
                total_loss += loss.item() * noisy_patches.size(0)
            # Calculate means
            mean_psnr_noisy = total_psnr_noisy / (num_batches * self.test_loader.batch_size)
            mean_psnr_denoised = total_psnr_denoised / (num_batches * self.test_loader.batch_size)

        return total_loss / len(self.test_loader.dataset),mean_psnr_noisy, mean_psnr_denoised


    def _select_and_update(self):
        # Perform sample selection based on the specified criterion
        if self.selection_criterion == 'random':
            selected_indices = self._random_sampling()
        elif self.selection_criterion == 'uncertainty':
            selected_indices = self._uncertainty_sampling()
        elif self.selection_criterion == 'entropy':
            selected_indices = self._entropy_sampling()
        elif self.selection_criterion == 'diversity':
            selected_indices = self._diversity_sampling()
        elif self.selection_criterion == 'diversity_2':
            selected_indices = self._diversity_sampling_2()
        elif self.selection_criterion == 'margin':
            selected_indices = self._margin_sampling()
        elif self.selection_criterion == 'gradient':
            selected_indices = self._gradient_magnitude_sampling()
        elif self.selection_criterion == 'noise_pattern_diversity':
            selected_indices = self._noise_pattern_diversity_sampling()
        elif self.selection_criterion == 'noise_pattern_diversity_2':
            selected_indices = self._noise_pattern_diversity_sampling_2()
        elif self.selection_criterion == 'noise_pattern_diversity_ablation':
            selected_indices = self._noise_pattern_diversity_sampling_ablation()
        elif self.selection_criterion == 'noise_pattern_diversity_ablation_2':
            selected_indices = self._noise_pattern_diversity_sampling_ablation_2()



        else:
            raise ValueError(f"Unknown selection criterion: {self.selection_criterion}")

        # Store newly added indices
        self.newly_added_indices = selected_indices
        # Add selected samples to the training set
        self.train_indices = np.concatenate([self.train_indices, selected_indices])
        # Remove selected samples from available pool
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_indices)

    def get_pool_loader(self):
        pool_subset = torch.utils.data.Subset(self.dataset, self.available_pool_indices)
        pool_loader = DataLoader(pool_subset, batch_size=self.batch_size, shuffle=False,
                                #  worker_init_fn=self._worker_init_fn
                                 )
        return pool_loader

    def _random_sampling(self):
        # Randomly select new samples from the available pool
        return np.random.choice(self.available_pool_indices, self.budget_per_iter, replace=False)

    def _uncertainty_sampling(self):
        # Evaluate uncertainty (MSE error) on available pool to select most uncertain samples
        # start_time = time.time()
        pool_loader = self.get_pool_loader()
        # end_time = time.time()
        # print(f"Time taken to create pool_loader: {end_time - start_time} seconds")
        uncertainties = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (noisy_patches, clean_patches) in enumerate(pool_loader):
              
                noisy_patches, clean_patches = noisy_patches.to(self.device), clean_patches.to(self.device)
                outputs = self.model(noisy_patches)
                mse_errors = ((outputs - clean_patches) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()  # Per-patch MSE
                uncertainties.extend(zip(mse_errors, range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)))
                
        # Sort by highest MSE (uncertainty) and select top indices
        
        uncertainties.sort(reverse=True, key=lambda x: x[0])
        selected_relative_indices = [idx for _, idx in uncertainties[:self.budget_per_iter]]
        selected_absolute_indices = [self.available_pool_indices[idx] for idx in selected_relative_indices]
      
        return np.array(selected_absolute_indices)

    def _entropy_sampling(self):
        pool_loader = self.get_pool_loader()
        entropy_values = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (noisy_patches, _) in enumerate(pool_loader):
                noisy_patches = noisy_patches.to(self.device)
                outputs = self.model(noisy_patches)
                # Compute entropy per patch
                output_probs = torch.sigmoid(outputs)  # Assumes sigmoid activation for denoising
                pixelwise_entropy = -output_probs * torch.log(output_probs + 1e-10)
                avg_entropy = pixelwise_entropy.mean(dim=[1, 2, 3]).cpu().numpy()
                entropy_values.extend(zip(avg_entropy, range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)))

        entropy_values.sort(reverse=True, key=lambda x: x[0])
        selected_relative_indices = [idx for _, idx in entropy_values[:self.budget_per_iter]]
        return np.array([self.available_pool_indices[idx] for idx in selected_relative_indices])

    def _diversity_sampling(self):
        
        pool_loader = self.get_pool_loader()
        features = []

        self.model.eval()
        with torch.no_grad():
            for noisy_patches, _ in pool_loader:
                noisy_patches = noisy_patches.to(self.device)
                # Use extract_features to get flattened feature embeddings
                feature_batch = self.model.extract_features(noisy_patches,extract_type='mean')
                features.append(feature_batch.cpu().numpy())

        # Combine feature batches into a single array
        features = np.concatenate(features)

        kmeans = KMeans(n_clusters=self.budget_per_iter, random_state=self.random_seed).fit(features)


        # Choose one sample per cluster by selecting closest point to each cluster center
        selected_indices = []
        for i in range(self.budget_per_iter):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            centroid = kmeans.cluster_centers_[i]
            closest_index = cluster_indices[np.argmin(np.linalg.norm(features[cluster_indices] - centroid, axis=1))]
            selected_indices.append(self.available_pool_indices[closest_index])

        return np.array(selected_indices)

    def _diversity_sampling_2(self):
        pool_loader = self.get_pool_loader()
        features = []

        self.model.eval()
        with torch.no_grad():
            for noisy_patches, _ in pool_loader:
                noisy_patches = noisy_patches.to(self.device)
                # Use extract_features to get flattened feature embeddings
                feature_batch = self.model.extract_features(noisy_patches,extract_type='flatten')
                features.append(feature_batch.cpu().numpy())

        # Combine feature batches into a single array
        features = np.concatenate(features)
        pca = PCA(n_components=300, random_state=self.random_seed)  # Adjust n_components based on desired tradeoff
        reduced_features = pca.fit_transform(features)  # Shape: [total_samples_in_pool, 300]

        print("Performing K-means clustering...")
        # Perform K-means on reduced features
        kmeans = KMeans(n_clusters=self.budget_per_iter, random_state=self.random_seed).fit(reduced_features)

        features = reduced_features

        # kmeans = KMeans(n_clusters=self.budget_per_iter, random_state=42).fit(features)
        print("Clustering completed.")
        # Choose one sample per cluster by selecting closest point to each cluster center
        selected_indices = []
        for i in range(self.budget_per_iter):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            centroid = kmeans.cluster_centers_[i]
            closest_index = cluster_indices[np.argmin(np.linalg.norm(features[cluster_indices] - centroid, axis=1))]
            selected_indices.append(self.available_pool_indices[closest_index])

        return np.array(selected_indices)


    def _margin_sampling(self):
        pool_loader = self.get_pool_loader()
        margin_values = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (noisy_patches, clean_patches) in enumerate(pool_loader):
                noisy_patches, clean_patches = noisy_patches.to(self.device), clean_patches.to(self.device)
                outputs = self.model(noisy_patches)
                # Calculate margin by comparing model output to a threshold (e.g., median pixel value)
                # Reshape clean patches to calculate median across dimensions 1, 2, and 3
                clean_patches_flattened = clean_patches.view(clean_patches.size(0), -1)
                median_val = clean_patches_flattened.median(dim=1).values

                # Reshape the median to match original dimensions
                median_val = median_val[:, None, None, None]

                margin_error = torch.abs(outputs.mean(dim=[1, 2, 3]) - median_val.squeeze())
                # median_val = clean_patches.median(dim=[1, 2, 3])
                # margin_error = torch.abs(outputs.mean(dim=[1, 2, 3]) - median_val)
                margin_values.extend(zip(margin_error.cpu().numpy(), range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)))

        margin_values.sort(reverse=True, key=lambda x: x[0])
        selected_relative_indices = [idx for _, idx in margin_values[:self.budget_per_iter]]
        return np.array([self.available_pool_indices[idx] for idx in selected_relative_indices])

    def _gradient_magnitude_sampling(self):
        """
        Select samples from the available pool based on gradient magnitude with respect to input.
        """
        pool_loader = self.get_pool_loader()
        gradient_magnitudes = []

        self.model.eval()
        # with torch.no_grad():
        for batch_idx, (noisy_patches, clean_patches) in enumerate(pool_loader):
            noisy_patches, clean_patches = noisy_patches.to(self.device), clean_patches.to(self.device)

            # Enable gradient calculation for the input
            noisy_patches.requires_grad = True

            # Forward pass
            outputs = self.model(noisy_patches)

            # Compute the loss
            loss = self.criterion(outputs, clean_patches)

            # Backward pass to compute gradients with respect to the input
            loss.backward()

            # Compute gradient magnitude for each sample in the batch
            batch_gradients = noisy_patches.grad.view(noisy_patches.size(0), -1)  # Flatten gradients per patch
            batch_magnitudes = torch.norm(batch_gradients, dim=1)  # Compute L2 norm of gradients

            # Store gradient magnitudes along with sample indices
            gradient_magnitudes.extend(zip(batch_magnitudes.cpu().numpy(),
                                            range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)))

            # Detach gradients to avoid memory leak
            noisy_patches.grad = None

        # Sort by largest gradient magnitude
        gradient_magnitudes.sort(reverse=True, key=lambda x: x[0])

        # Select top samples based on gradient magnitude
        selected_relative_indices = [idx for _, idx in gradient_magnitudes[:self.budget_per_iter]]
        selected_absolute_indices = [self.available_pool_indices[idx] for idx in selected_relative_indices]

        return np.array(selected_absolute_indices)

    def _noise_pattern_diversity_sampling(self):
        """
        A novel sampling strategy that selects patches based on their noise pattern characteristics.
        The method:
        1. Extracts noise patterns by subtracting clean from noisy images
        2. Characterizes each noise pattern using multiple metrics
        3. Clusters these characteristics and selects diverse representatives
        """
        pool_loader = self.get_pool_loader()
        noise_characteristics = []

        with torch.no_grad():
            for noisy_patches, clean_patches in pool_loader:
                noisy_patches = noisy_patches.to(self.device)
                clean_patches = clean_patches.to(self.device)

                # Extract noise patterns
                noise = noisy_patches - clean_patches
                # Model prediction
                outputs = self.model(noisy_patches)

                # Calculate various characteristics for each patch
                for i in range(noise.size(0)):
                    patch_noise = noise[i]

                    # Model-related features
                    # 1. Reconstruction Error
                    reconstruction_error = torch.nn.functional.mse_loss(
                        outputs[i],
                        clean_patches[i]
                    ).item()

                    # 2. Model Prediction Uncertainty
                    # Use the difference between noisy input and model output as an uncertainty measure
                    model_uncertainty = torch.abs(noisy_patches[i] - outputs[i]).mean().item()

                    # 3. Prediction Variance
                    prediction_variance = torch.var(outputs[i]).item()
                    ####################################################
                    # 1. Directional gradients (horizontal and vertical)
                    dx = torch.abs(patch_noise[:, :, 1:] - patch_noise[:, :, :-1]).mean()
                    dy = torch.abs(patch_noise[:, 1:, :] - patch_noise[:, :-1, :]).mean()

                    # 2. Local variance in different regions
                    patch_size = patch_noise.size(-1)
                    quadrant_size = patch_size // 2
                    quadrant_variances = [
                        torch.var(patch_noise[:, :quadrant_size, :quadrant_size]),
                        torch.var(patch_noise[:, :quadrant_size, quadrant_size:]),
                        torch.var(patch_noise[:, quadrant_size:, :quadrant_size]),
                        torch.var(patch_noise[:, quadrant_size:, quadrant_size:])
                    ]

                    # 3. Frequency characteristics (using DCT-like approach)
                    high_freq_content = torch.abs(patch_noise[:, 1:, 1:] +
                                              patch_noise[:, :-1, :-1] -
                                              patch_noise[:, 1:, :-1] -
                                              patch_noise[:, :-1, 1:]).mean()

                    # 4. Noise intensity distribution
                    noise_hist = torch.histc(patch_noise, bins=10)
                    entropy = -torch.sum(noise_hist * torch.log(noise_hist + 1e-10))

                    # 5. Spatial correlation
                    center_mean = patch_noise[:,
                                          quadrant_size-2:quadrant_size+2,
                                          quadrant_size-2:quadrant_size+2].mean()
                    edge_mean = torch.cat([
                        patch_noise[:, :2, :].flatten(),
                        patch_noise[:, -2:, :].flatten(),
                        patch_noise[:, :, :2].flatten(),
                        patch_noise[:, :, -2:].flatten()
                    ]).mean()
                    spatial_correlation = torch.abs(center_mean - edge_mean)

                    

                    # Combine all characteristics into a feature vector
                    features = torch.tensor([
                        reconstruction_error,
                        model_uncertainty,
                        prediction_variance,
                        dx.item(),
                        dy.item(),
                        *[v.item() for v in quadrant_variances],
                        high_freq_content.item(),
                        entropy.item(),
                        spatial_correlation.item(),
                        
                    ])

                    noise_characteristics.append((features, len(noise_characteristics)))

        # Convert to numpy for clustering
        features = np.array([f[0].cpu().numpy() for f in noise_characteristics])

        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        # Use DBSCAN for adaptive clustering
        eps = np.percentile(pdist(features), 10)  # Use 10th percentile of pairwise distances
        dbscan = DBSCAN(eps=eps, min_samples=2)
        clusters = dbscan.fit_predict(features)

        # Select samples ensuring diversity
        selected_indices = []
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Distribute budget across clusters and noise points
        noise_points = np.where(clusters == -1)[0]
        n_noise_selections = min(self.budget_per_iter // 3, len(noise_points))
        cluster_selections = self.budget_per_iter - n_noise_selections

        # Select noise points (they might represent unique patterns)
        if n_noise_selections > 0:
            noise_distances = pdist(features[noise_points])
            noise_indices = noise_points[
                np.random.choice(len(noise_points),
                              size=n_noise_selections,
                              replace=False)
            ]
            selected_indices.extend([noise_characteristics[idx][1] for idx in noise_indices])

        # Select from clusters
        if n_clusters > 0:
            selections_per_cluster = max(1, cluster_selections // n_clusters)
            for cluster_id in range(n_clusters):
                cluster_points = np.where(clusters == cluster_id)[0]
                if len(cluster_points) > 0:
                    # Select points that are most different within each cluster
                    cluster_features = features[cluster_points]
                    distances = pdist(cluster_features)
                    if len(distances) > 0:
                        diverse_indices = cluster_points[
                            np.random.choice(len(cluster_points),
                                          size=min(selections_per_cluster, len(cluster_points)),
                                          replace=False)
                        ]
                        selected_indices.extend([noise_characteristics[idx][1] for idx in diverse_indices])

        # If we haven't filled our budget, add random selections
        while len(selected_indices) < self.budget_per_iter:
            remaining = self.budget_per_iter - len(selected_indices)
            available = list(set(range(len(noise_characteristics))) - set(selected_indices))
            if not available:
                break
            selected_indices.extend(np.random.choice(available, size=min(remaining, len(available)), replace=False))

        # Convert to absolute indices
        return np.array([self.available_pool_indices[idx] for idx in selected_indices[:self.budget_per_iter]])

    def _noise_pattern_diversity_sampling_2(self):
        """
        Enhanced noise pattern diversity sampling strategy that leverages the model's
        understanding of noise characteristics.

        Steps:
        1. Predict noise patterns using the current model
        2. Extract multi-dimensional noise characteristics
        3. Cluster and select diverse representatives
        """
        pool_loader = self.get_pool_loader()
        noise_characteristics = []

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for noisy_patches, clean_patches in pool_loader:
                noisy_patches = noisy_patches.to(self.device)
                clean_patches = clean_patches.to(self.device)

                # Model-based noise prediction
                predicted_clean_patches = self.model(noisy_patches)

                # Extract noise patterns
                # model_noise = noisy_patches - predicted_clean_patches
                model_noise = clean_patches - predicted_clean_patches
                ground_truth_noise = noisy_patches - clean_patches

                for i in range(noisy_patches.size(0)):
                    # Model's predicted noise pattern
                    patch_model_noise = model_noise[i]
                    # Ground truth noise pattern
                    patch_ground_truth_noise = ground_truth_noise[i]

                    # Model Uncertainty Features
                    # 1. Reconstruction Error
                    reconstruction_error = torch.nn.functional.mse_loss(
                        predicted_clean_patches[i],
                        clean_patches[i]
                    ).item()

                    # 2. Model Prediction Variance
                    prediction_variance = torch.var(predicted_clean_patches[i]).item()

                    # 3. Noise Pattern Characteristics
                    # Directional gradients
                    dx_model = torch.abs(patch_model_noise[:, :, 1:] - patch_model_noise[:, :, :-1]).mean().item()
                    dy_model = torch.abs(patch_model_noise[:, 1:, :] - patch_model_noise[:, :-1, :]).mean().item()

                    # 4. Difference between model noise and ground truth noise
                    noise_prediction_difference = torch.abs(patch_model_noise - patch_ground_truth_noise).mean().item()

                    # 5. Spectral characteristics of model noise
                    fft_model_noise = torch.fft.fft2(patch_model_noise)
                    spectral_energy = torch.abs(fft_model_noise).mean().item()

                    # Combine features
                    features = torch.tensor([
                        reconstruction_error,
                        prediction_variance,
                        dx_model,
                        dy_model,
                        noise_prediction_difference,
                        spectral_energy
                    ])

                    # Store features along with original index
                    noise_characteristics.append((features, len(noise_characteristics)))

        # Convert to numpy for clustering
        features = np.array([f[0].cpu().numpy() for f in noise_characteristics])

        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        # Clustering
        eps = np.percentile(pdist(features), 10)
        dbscan = DBSCAN(eps=eps, min_samples=2)
        clusters = dbscan.fit_predict(features)

        # Sample selection logic (similar to previous implementation)
        selected_indices = []
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Distribution of selections
        noise_points = np.where(clusters == -1)[0]
        n_noise_selections = min(self.budget_per_iter // 3, len(noise_points))
        cluster_selections = self.budget_per_iter - n_noise_selections

        # Select noise points
        if n_noise_selections > 0:
            noise_indices = noise_points[
                np.random.choice(len(noise_points),
                              size=n_noise_selections,
                              replace=False)
            ]
            selected_indices.extend([noise_characteristics[idx][1] for idx in noise_indices])

        # Select from clusters
        if n_clusters > 0:
            selections_per_cluster = max(1, cluster_selections // n_clusters)
            for cluster_id in range(n_clusters):
                cluster_points = np.where(clusters == cluster_id)[0]
                if len(cluster_points) > 0:
                    diverse_indices = cluster_points[
                        np.random.choice(len(cluster_points),
                                      size=min(selections_per_cluster, len(cluster_points)),
                                      replace=False)
                    ]
                    selected_indices.extend([noise_characteristics[idx][1] for idx in diverse_indices])

        # Fill remaining budget if needed
        while len(selected_indices) < self.budget_per_iter:
            remaining = self.budget_per_iter - len(selected_indices)
            available = list(set(range(len(noise_characteristics))) - set(selected_indices))
            if not available:
                break
            selected_indices.extend(np.random.choice(available, size=min(remaining, len(available)), replace=False))

        # Convert to absolute indices
        return np.array([self.available_pool_indices[idx] for idx in selected_indices[:self.budget_per_iter]])

    def _noise_pattern_diversity_sampling_ablation(self, ablation_components=[]):
        """
        Modified sampling strategy for ablation study.
        Args:
        - ablation_components: List of feature names to exclude in the ablation study.
        """
        pool_loader = self.get_pool_loader()
        noise_characteristics = []

        with torch.no_grad():
            for noisy_patches, clean_patches in pool_loader:
                noisy_patches = noisy_patches.to(self.device)
                clean_patches = clean_patches.to(self.device)

                noise = noisy_patches - clean_patches
                outputs = self.model(noisy_patches)

                for i in range(noise.size(0)):
                    patch_noise = noise[i]

                    # Extract individual characteristics based on ablation
                    features = []

                    if 'reconstruction_error' not in ablation_components:
                        reconstruction_error = torch.nn.functional.mse_loss(outputs[i], clean_patches[i]).item()
                        features.append(reconstruction_error)

                    if 'model_uncertainty' not in ablation_components:
                        model_uncertainty = torch.abs(noisy_patches[i] - outputs[i]).mean().item()
                        features.append(model_uncertainty)

                    if 'prediction_variance' not in ablation_components:
                        prediction_variance = torch.var(outputs[i]).item()
                        features.append(prediction_variance)

                    if 'dx' not in ablation_components:
                        dx = torch.abs(patch_noise[:, :, 1:] - patch_noise[:, :, :-1]).mean().item()
                        features.append(dx)

                    if 'dy' not in ablation_components:
                        dy = torch.abs(patch_noise[:, 1:, :] - patch_noise[:, :-1, :]).mean().item()
                        features.append(dy)

                    if 'quadrant_variance' not in ablation_components:
                        quadrant_size = patch_noise.size(-1) // 2
                        quadrant_variances = [
                            torch.var(patch_noise[:, :quadrant_size, :quadrant_size]).item(),
                            torch.var(patch_noise[:, :quadrant_size, quadrant_size:]).item(),
                            torch.var(patch_noise[:, quadrant_size:, :quadrant_size]).item(),
                            torch.var(patch_noise[:, quadrant_size:, quadrant_size:]).item(),
                        ]
                        features.extend(quadrant_variances)

                    if 'high_freq_content' not in ablation_components:
                        high_freq_content = torch.abs(
                            patch_noise[:, 1:, 1:] + patch_noise[:, :-1, :-1] -
                            patch_noise[:, 1:, :-1] - patch_noise[:, :-1, 1:]
                        ).mean().item()
                        features.append(high_freq_content)

                    if 'entropy' not in ablation_components:
                        noise_hist = torch.histc(patch_noise, bins=10)
                        entropy = -torch.sum(noise_hist * torch.log(noise_hist + 1e-10)).item()
                        features.append(entropy)

                    if 'spatial_correlation' not in ablation_components:
                        center_mean = patch_noise[:,
                                          quadrant_size-2:quadrant_size+2,
                                          quadrant_size-2:quadrant_size+2].mean()
                        edge_mean = torch.cat([
                            patch_noise[:, :2, :].flatten(),
                            patch_noise[:, -2:, :].flatten(),
                            patch_noise[:, :, :2].flatten(),
                            patch_noise[:, :, -2:].flatten()
                        ]).mean()
                        spatial_correlation = torch.abs(center_mean - edge_mean)
                        features.append(spatial_correlation.item())

                    # Collect features for this patch
                    noise_characteristics.append((torch.tensor(features), len(noise_characteristics)))

            # Convert to numpy for clustering
            features = np.array([f[0].cpu().numpy() for f in noise_characteristics])

            # Normalize features
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

            # Use DBSCAN for adaptive clustering
            eps = np.percentile(pdist(features), 10)  # Use 10th percentile of pairwise distances
            dbscan = DBSCAN(eps=eps, min_samples=2)
            clusters = dbscan.fit_predict(features)

            # Select samples ensuring diversity
            selected_indices = []
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

            # Distribute budget across clusters and noise points
            noise_points = np.where(clusters == -1)[0]
            n_noise_selections = min(self.budget_per_iter // 3, len(noise_points))
            cluster_selections = self.budget_per_iter - n_noise_selections

            # Select noise points (they might represent unique patterns)
            if n_noise_selections > 0:
                noise_distances = pdist(features[noise_points])
                noise_indices = noise_points[
                    np.random.choice(len(noise_points),
                                  size=n_noise_selections,
                                  replace=False)
                ]
                selected_indices.extend([noise_characteristics[idx][1] for idx in noise_indices])

            # Select from clusters
            if n_clusters > 0:
                selections_per_cluster = max(1, cluster_selections // n_clusters)
                for cluster_id in range(n_clusters):
                    cluster_points = np.where(clusters == cluster_id)[0]
                    if len(cluster_points) > 0:
                        # Select points that are most different within each cluster
                        cluster_features = features[cluster_points]
                        distances = pdist(cluster_features)
                        if len(distances) > 0:
                            diverse_indices = cluster_points[
                                np.random.choice(len(cluster_points),
                                              size=min(selections_per_cluster, len(cluster_points)),
                                              replace=False)
                            ]
                            selected_indices.extend([noise_characteristics[idx][1] for idx in diverse_indices])

            # If we haven't filled our budget, add random selections
            while len(selected_indices) < self.budget_per_iter:
                remaining = self.budget_per_iter - len(selected_indices)
                available = list(set(range(len(noise_characteristics))) - set(selected_indices))
                if not available:
                    break
                selected_indices.extend(np.random.choice(available, size=min(remaining, len(available)), replace=False))

            # Convert to absolute indices
            return np.array([self.available_pool_indices[idx] for idx in selected_indices[:self.budget_per_iter]])
    def _noise_pattern_diversity_sampling_ablation_2(self, ablation_components=[]):
        """
        Enhanced noise pattern diversity sampling strategy that leverages the model's
        understanding of noise characteristics.

        Steps:
        1. Predict noise patterns using the current model
        2. Extract multi-dimensional noise characteristics
        3. Cluster and select diverse representatives
        """
        pool_loader = self.get_pool_loader()
        noise_characteristics = []

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for noisy_patches, clean_patches in pool_loader:
                noisy_patches = noisy_patches.to(self.device)
                clean_patches = clean_patches.to(self.device)

                # Model-based noise prediction
                predicted_clean_patches = self.model(noisy_patches)

                # Extract noise patterns
                # model_noise = noisy_patches - predicted_clean_patches
                model_noise = clean_patches - predicted_clean_patches
                ground_truth_noise = noisy_patches - clean_patches

                for i in range(noisy_patches.size(0)):
                    # Model's predicted noise pattern
                    patch_model_noise = model_noise[i]
                    # Ground truth noise pattern
                    patch_ground_truth_noise = ground_truth_noise[i]

                    # Model Uncertainty Features
                    # 1. Reconstruction Error
                    if 'reconstruction_error' not in ablation_components:
                      reconstruction_error = torch.nn.functional.mse_loss(
                          predicted_clean_patches[i],
                          clean_patches[i]
                      ).item()

                    # 2. Model Prediction Variance
                    if 'prediction_variance' not in ablation_components:
                      prediction_variance = torch.var(predicted_clean_patches[i]).item()

                    # 3. Noise Pattern Characteristics
                    # Directional gradients
                    if 'dx' not in ablation_components:
                      dx_model = torch.abs(patch_model_noise[:, :, 1:] - patch_model_noise[:, :, :-1]).mean().item()
                    if 'dy' not in ablation_components:
                      dy_model = torch.abs(patch_model_noise[:, 1:, :] - patch_model_noise[:, :-1, :]).mean().item()

                    # 4. Difference between model noise and ground truth noise
                    if 'noise_prediction_difference' not in ablation_components:
                      noise_prediction_difference = torch.abs(patch_model_noise - patch_ground_truth_noise).mean().item()

                    # 5. Spectral characteristics of model noise
                    if 'spectral_energy' not in ablation_components:
                      fft_model_noise = torch.fft.fft2(patch_model_noise)
                      spectral_energy = torch.abs(fft_model_noise).mean().item()

                    # Combine features
                    features = torch.tensor([
                        reconstruction_error,
                        prediction_variance,
                        dx_model,
                        dy_model,
                        noise_prediction_difference,
                        spectral_energy
                    ])

                    # Store features along with original index
                    noise_characteristics.append((features, len(noise_characteristics)))

        # Convert to numpy for clustering
        features = np.array([f[0].cpu().numpy() for f in noise_characteristics])

        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        # Clustering
        eps = np.percentile(pdist(features), 10)
        dbscan = DBSCAN(eps=eps, min_samples=2)
        clusters = dbscan.fit_predict(features)

        # Sample selection logic (similar to previous implementation)
        selected_indices = []
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Distribution of selections
        noise_points = np.where(clusters == -1)[0]
        n_noise_selections = min(self.budget_per_iter // 3, len(noise_points))
        cluster_selections = self.budget_per_iter - n_noise_selections

        # Select noise points
        if n_noise_selections > 0:
            noise_indices = noise_points[
                np.random.choice(len(noise_points),
                              size=n_noise_selections,
                              replace=False)
            ]
            selected_indices.extend([noise_characteristics[idx][1] for idx in noise_indices])

        # Select from clusters
        if n_clusters > 0:
            selections_per_cluster = max(1, cluster_selections // n_clusters)
            for cluster_id in range(n_clusters):
                cluster_points = np.where(clusters == cluster_id)[0]
                if len(cluster_points) > 0:
                    diverse_indices = cluster_points[
                        np.random.choice(len(cluster_points),
                                      size=min(selections_per_cluster, len(cluster_points)),
                                      replace=False)
                    ]
                    selected_indices.extend([noise_characteristics[idx][1] for idx in diverse_indices])

        # Fill remaining budget if needed
        while len(selected_indices) < self.budget_per_iter:
            remaining = self.budget_per_iter - len(selected_indices)
            available = list(set(range(len(noise_characteristics))) - set(selected_indices))
            if not available:
                break
            selected_indices.extend(np.random.choice(available, size=min(remaining, len(available)), replace=False))

        # Convert to absolute indices
        return np.array([self.available_pool_indices[idx] for idx in selected_indices[:self.budget_per_iter]])




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
print(f"Max pixel value: {max_pixel}")




# use example
model = LightRIDNet().to(device)
pipeline = ActiveLearningPipeline(
    model=model,
    train_indices=initial_train_indices,
    test_indices=test_indices,
    available_pool_indices=available_pool_indices,
    selection_criterion='noise_pattern_diversity',
    iterations=15,
    budget_per_iter=128,
    dataset=sample_dataset,
    batch_size=batch_size,
    max_pixel=max_pixel,
    random_seed=random_seed,
    train_all=True
)
# Run active learning pipeline
mse_scores, psnr_improvment_scores = pipeline.run_pipeline()
print("MSE Scores at each iteration:", mse_scores)
print("PSNR Improvement Scores at each iteration:", psnr_improvment_scores)