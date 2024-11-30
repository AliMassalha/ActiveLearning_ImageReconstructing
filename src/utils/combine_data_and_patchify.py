import os
import pandas as pd
import glob

def combining_data():


    base_path = 'C:/Users/My/Desktop/ActiveLearning_ImageReconstructing/'
    sidd_path = 'data/SIDD_Medium_Srgb/Data'
    renoir_path = 'data/RENOIR'
    polyu_path = 'data/PolyU_PairedImages'

    # List to store paths for each dataset
    data = []

    # 1. Gather SIDD "pair1" data
    for folder in os.listdir(sidd_path):
        if "pair1" in folder:
            folder_path = os.path.join(sidd_path, folder)
            gt_image = glob.glob(os.path.join(folder_path, '*GT*.PNG'))[0]
            noisy_image = glob.glob(os.path.join(folder_path, '*NOISY*.PNG'))[0]
            data.append({'dataset': 'SIDD', 'gt_path': gt_image, 'noisy_path': noisy_image})

    # 2. Gather RENOIR data
    camera_types = ['Mi3_Aligned', 'T3i_Aligned', 'S90_Aligned']
    for camera in camera_types:
        camera_path = os.path.join(renoir_path, camera)
        for batch_folder in os.listdir(camera_path):
            batch_path = os.path.join(camera_path, batch_folder)
            if os.path.isdir(batch_path):
                reference_files = glob.glob(os.path.join(batch_path, '*Reference.bmp'))
                noisy_files = glob.glob(os.path.join(batch_path, '*Noisy.bmp'))
                for ref, noisy in zip(reference_files, noisy_files):
                    data.append({'dataset': 'RENOIR', 'gt_path': ref, 'noisy_path': noisy})


    # 3. Gather PolyU data
    poly_count = 0
    for folder in os.listdir(polyu_path):
        folder_path = os.path.join(polyu_path, folder)
        poly_count = 0
        for pair in os.listdir(folder_path):
            if poly_count >= 2:
                break
            pair_path = os.path.join(folder_path, pair)
            gt_image = glob.glob(os.path.join(pair_path, '*mean*'))[0]
            noisy_image = glob.glob(os.path.join(pair_path, '*noisy*'))[0]
            data.append({'dataset': 'PolyU', 'gt_path': gt_image, 'noisy_path': noisy_image})
            poly_count += 1

    df = pd.DataFrame(data)
    
    return df
    


import os
import cv2
import numpy as np
from patchify import patchify
from PIL import Image


def save_image_patches(df, output_base_path, patch_size=256):
    """
    Save image patches as PNG files (good balance between quality and size)
    
    Structure:
    output_base_path/
    ├── clean/
    │   ├── dataset_name_imageid_patchid.png
    │   └── ...
    ├── noisy/
    │   ├── dataset_name_imageid_patchid.png
    │   └── ...
    └── metadata.json
    """
    
    def process_and_save_patches(img_path, output_dir, dataset_name, image_id):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
        
        # Reshape patches to be 4D: (num_patches, height, width, channels)
        patches = patches.reshape(-1, patch_size, patch_size, 3)
        
        saved_paths = []
        # Save each patch as PNG
        for patch_id, patch in enumerate(patches):
            filename = f"{dataset_name}_{image_id:04d}_{patch_id:04d}.png"
            save_path = os.path.join(output_dir, filename)
            
            # Convert to PIL Image and save
            Image.fromarray(patch).save(
                save_path, 
                format='PNG',
                optimize=True  # Enable compression optimization
            )
            saved_paths.append(save_path)
            
        return len(patches)

    # Create directory structure
    clean_dir = os.path.join(output_base_path, 'clean')
    noisy_dir = os.path.join(output_base_path, 'noisy')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    # Process each image pair
    metadata = []
    for idx, row in df.iterrows():
        dataset_name = row['dataset'].lower()
        
        # Save clean and noisy patches
        num_patches = process_and_save_patches(
            row['gt_path'], clean_dir, dataset_name, idx)
        process_and_save_patches(
            row['noisy_path'], noisy_dir, dataset_name, idx)
        


def compare_storage_formats(sample_patch):
    """
    Compare storage sizes of different formats for a single patch
    """
    # Create a temporary directory
    temp_dir = "temp_format_comparison"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save in different formats
    formats = {
        'NPY': ('.npy', lambda x, p: np.save(p, x)),
        'PNG': ('.png', lambda x, p: Image.fromarray(x).save(p, format='PNG', optimize=True)),
        'JPEG': ('.jpg', lambda x, p: Image.fromarray(x).save(p, format='JPEG', quality=95)),
    }
    
    sizes = {}
    for format_name, (ext, save_func) in formats.items():
        path = os.path.join(temp_dir, f"test{ext}")
        save_func(sample_patch, path)
        sizes[format_name] = os.path.getsize(path)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    return sizes



# Example usage and format comparison:
df = combining_data()
output_path = 'data/patches'

# First, let's compare storage formats with a sample patch
path = df['gt_path'].iloc[0]
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
patches = patchify(img, (256, 256, 3), step=256)
sample_patch = patches[0, 0, 0]

# Compare sizes
sizes = compare_storage_formats(sample_patch)
for format_name, size in sizes.items():
    print(f"{format_name}: {size/1024:.2f} KB")

# Save all patches
save_image_patches(df, output_path)

