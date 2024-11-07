import os
import shutil
import random

# Define paths
source_folder = 'data/PolyUDataset'
destination_folder = 'data/PolyU_PairedImages'
sample_size = 15  # Number of images to sample per scene

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Iterate through each scene folder in the source folder
for scene_folder in os.listdir(source_folder):
    scene_path = os.path.join(source_folder, scene_folder)
    
    # Skip if it's not a directory
    if not os.path.isdir(scene_path):
        continue
    
    # Identify the ground truth image and noisy images
    ground_truth_image = None
    noisy_images = []
    
    for img_file in os.listdir(scene_path):
        img_path = os.path.join(scene_path, img_file)
        if 'mean' in img_file:
            ground_truth_image = img_path
        else:
            noisy_images.append(img_path)
    
    # Select a random sample of noisy images
    sampled_images = random.sample(noisy_images, min(sample_size, len(noisy_images)))
    
    # Create paired subfolders for each sampled noisy image
    for idx, noisy_image in enumerate(sampled_images, start=1):
        paired_folder = os.path.join(destination_folder, scene_folder, f"pair_{idx}")
        os.makedirs(paired_folder, exist_ok=True)
        
        # Copy noisy and ground truth images into the paired folder
        noisy_image_name = f"noisy.jpg" 
        ground_truth_name = f"mean.jpg"  
        
        shutil.copy(noisy_image, os.path.join(paired_folder, noisy_image_name))
        shutil.copy(ground_truth_image, os.path.join(paired_folder, ground_truth_name))

print("Random sampled paired image folders have been successfully created.")
