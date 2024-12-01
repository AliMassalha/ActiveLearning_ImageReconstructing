Here's a **README.md** file based on your instructions:

---

# **Active Learning for Image Denoising**

This repository implements an active learning pipeline for image denoising using novel noise pattern diversity sampling methods. It also includes traditional baselines and experiments inspired by the paper *"On-Demand Learning for Image Reconstruction."*

## **Setup**

1. **Download and Extract the Data**  
   Before running any experiments, ensure that the necessary datasets are downloaded and extracted. Run the `download_data.py` script:
   ```bash
   python download_data.py
   ```
   This will download and extract the clean and noisy image patches into the `data/patches/` directory.

---

## **Running Experiments**

All experimental evaluations are located in the `src/evaluation/` folder. To run an experiment, simply execute the relevant file.

### Example:
```bash
python src/evaluation/<experiment_file>.py
```

- **`ablation_studies.py`**: Runs ablation studies to evaluate the contribution of each component in the custom sampling methods.
- **`budget_comparison.py`**: Compares the effect of different sampling budgets on model performance.
- **`compare_strategies.py`**: Compares the custom sampling strategies with traditional methods (random, uncertainty).
- **`noise_augmentation_comparison.py`**: Evaluates the robustness of the models with different types of noise augmentation.
- **`train_size_comparison.py`**: Assesses the impact of varying dataset sizes on model performance.

---

## **Training**

The `src/training/` folder contains files for training models on the dataset.

### **1. Training on Full Dataset**
To train both models (U-Net and LightRIDNet) on the entire dataset:
```bash
python src/training/train.py
```
This script trains the models using all available clean-noisy pairs and evaluates their performance.

### **2. Training with Active Learning Pipeline**
To run an example of the active learning pipeline:
```bash
python src/training/train_active_learning.py
```
This file demonstrates the pipeline, but it mainly serves as a reference implementation of the active learning code.

---

## **Other Training Files**
The remaining files in the `src/training/` folder are our attempts at reproducing and expanding upon the paper *"On-Demand Learning for Image Reconstruction"*:

- **`on_demand_learning_w_added_noise.py`**: Incorporates additional noise into the training pipeline as part of our custom experiments.
- **`on_demand_learning_wo_added_noise.py`**: Runs the on-demand learning framework without adding additional noise.

---

## **Contributors**
- **Ali Massalha** ([ali.m@campus.technion.ac.il](mailto:ali.m@campus.technion.ac.il))  
- **Adan Gaben** ([adangaben@campus.technion.ac.il](mailto:adangaben@campus.technion.ac.il))

---

## **References**
- *On-Demand Learning for Deep Image Restoration* ([Link to paper](https://arxiv.org/pdf/1612.01380))  
- *RIDNet: Residual Image Denoising Network* ([Link to paper](https://arxiv.org/pdf/1904.07396v2.pdf))  

