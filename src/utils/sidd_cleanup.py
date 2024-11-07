import os
import shutil

# Set the base path to your SIDD_Medium_Srgb/Data folder
base_path = 'data/SIDD_Medium_Srgb/Data'

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        # Get all image filenames in the current folder
        images = sorted([img for img in os.listdir(folder_path) if img.endswith('.PNG')])

        # Assuming each folder has exactly two pairs of GT and NOISY images
        pairs = [(images[i], images[i+2]) for i in range(0, len(images)//2)]
        # print(pairs)
        
        # Create new folders for each pair and move files
        for idx, (gt_image, noisy_image) in enumerate(pairs, 1):
            new_folder = f"{folder}_pair{idx}"
            new_folder_path = os.path.join(base_path, new_folder)
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Move each pair to the new folder
            shutil.move(os.path.join(folder_path, gt_image), os.path.join(new_folder_path, gt_image))
            shutil.move(os.path.join(folder_path, noisy_image), os.path.join(new_folder_path, noisy_image))

        # Remove the original folder after all pairs have been moved
        shutil.rmtree(folder_path)

print("Folders restructured successfully.")
