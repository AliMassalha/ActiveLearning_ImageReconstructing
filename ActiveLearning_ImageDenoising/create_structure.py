import os

# Define the project structure
project_structure = {
    "ActiveLearning_ImageDenoising": [
        "data/DIV2K/DIV2K_train_HR/",
        "data/DIV2K/DIV2K_train_LR_bicubic/",
        "data/DIV2K/DIV2K_train_LR_unknown/",
        "data/DIV2K/DIV2K_valid_HR/",
        "data/DIV2K/DIV2K_valid_LR_bicubic/",
        "data/DIV2K/DIV2K_valid_LR_unknown/",
        "data/other_datasets/",
        "src/models/",
        "src/active_learning/",
        "src/training/",
        "src/evaluation/",
        "src/utils/",
        "notebooks/",
        "results/models/",
        "results/logs/",
        "results/figures/",
        "results/reports/",
        "tests/",
        ".vscode/"
    ],
    "ActiveLearning_ImageDenoising_files": [
        "data/preprocess.py",
        "src/models/unet.py",
        "src/models/custom_model.py",
        "src/active_learning/random_sampling.py",
        "src/active_learning/uncertainty_sampling.py",
        "src/active_learning/diversity_sampling.py",
        "src/training/train.py",
        "src/training/train_active_learning.py",
        "src/evaluation/evaluate.py",
        "src/evaluation/visualization.py",
        "src/utils/data_loader.py",
        "src/utils/image_processing.py",
        "src/utils/checkpoint.py",
        "src/main.py",
        "notebooks/exploratory_data_analysis.ipynb",
        "notebooks/training_experiments.ipynb",
        "tests/test_models.py",
        "tests/test_active_learning.py",
        "tests/test_utils.py",
        ".vscode/launch.json",
        ".vscode/settings.json",
        ".vscode/tasks.json",
        "README.md",
        "requirements.txt",
        "setup.py",
        "LICENSE"
    ]
}

# Create directories
for folder in project_structure["ActiveLearning_ImageDenoising"]:
    os.makedirs(folder, exist_ok=True)

# Create files
for file_path in project_structure["ActiveLearning_ImageDenoising_files"]:
    with open(file_path, 'w') as f:
        pass  # Create an empty file

print("Project structure created successfully!")
