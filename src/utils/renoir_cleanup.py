import os
import glob
import shutil
from typing import List, Dict
import pandas as pd

class RenoirDatasetCleaner:
    def __init__(self, dataset_path: str):
        """
        Initialize the RENOIR dataset cleaner
        Args:
            dataset_path (str): Root directory of the RENOIR dataset
        """
        self.dataset_path = dataset_path
        self.camera_types = ['Mi3_Aligned', 'T3i_Aligned', 'S90_Aligned']
        self.extra_noisy_path = os.path.join(dataset_path, 'extra_noisy_files')
        os.makedirs(self.extra_noisy_path, exist_ok=True)
        
    def analyze_space(self) -> pd.DataFrame:
        """
        Analyze space usage by file types
        Returns DataFrame with space usage statistics
        """
        stats = []
        
        for camera in self.camera_types:
            camera_path = os.path.join(self.dataset_path, camera)
            if not os.path.exists(camera_path):
                continue
                
            for root, _, files in os.walk(camera_path):
                # print(root)
                for file in files:
                    # print(files)
                    if file == "Thumbs.db":
                        continue
                    file_path = os.path.join(root, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                    
                    file_type = 'other'
                    if 'Reference' in file:
                        file_type = 'reference'
                    elif 'Noisy' in file and file.endswith('.bmp'):
                        file_type = 'noisy'
                    elif 'full' in file.lower():
                        file_type = 'full'
                    elif 'mask' in file.lower():
                        file_type = 'mask'
                    elif file.endswith('.png'):
                        file_type = 'plot'
                        
                    stats.append({
                        'camera': camera,
                        'file_type': file_type,
                        'filename': file,
                        'size_mb': size_mb,
                        'path': file_path
                    })
        # print(stats)
        return pd.DataFrame.from_dict(stats)
    
    def identify_pairs(self) -> Dict[str, List[str]]:
        """
        Identify all reference-noisy pairs and their associated files
        Returns dict with reference files as keys and lists of related files as values
        """
        pairs = {}
        
        for camera in self.camera_types:
            camera_path = os.path.join(self.dataset_path, camera)
            if not os.path.exists(camera_path):
                continue
                
            # Find all reference images
            for root, _, files in os.walk(camera_path):
                reference_files = [f for f in files if 'Reference' in f and f.endswith('.bmp')]
                
                for ref_file in reference_files:
                    base_name = ref_file.replace('Reference.bmp', '')
                    ref_path = os.path.join(root, ref_file)
                    
                    # Find all related files
                    related_files = [
                        os.path.join(root, f) for f in files
                        if base_name in f and f != ref_file
                    ]
                    
                    pairs[ref_path] = related_files
                    
        return pairs
    
    def cleanup_dataset(self, keep_second_noisy: bool = False, 
                       move_deleted: bool = True) -> Dict[str, int]:
        """
        Clean up the dataset by removing unnecessary files
        Args:
            keep_second_noisy: Whether to keep second noisy image
            move_deleted: If True, move files to a 'deleted' folder instead of removing
        Returns:
            Dict with statistics about removed files
        """
        stats = {
            'removed_full': 0,
            'removed_mask': 0,
            'removed_plot': 0,
            'removed_second_noisy': 0,
            'space_saved_mb': 0
        }
        
        # Create deleted folder if needed
        if move_deleted:
            deleted_folder = os.path.join(self.dataset_path, 'deleted_files')
            os.makedirs(deleted_folder, exist_ok=True)
        
        # Analyze current space usage
        # df = self.analyze_space()
        
        for camera in self.camera_types:
            camera_path = os.path.join(self.dataset_path, camera)
            if not os.path.exists(camera_path):
                continue
                
            for root, _, files in os.walk(camera_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    should_remove = False
                    
                    # Check if file should be removed
                    if 'full' in file.lower():
                        should_remove = True
                        stats['removed_full'] += 1
                    elif 'mask' in file.lower():
                        should_remove = True
                        stats['removed_mask'] += 1
                    elif file.endswith('.png'):
                        should_remove = True
                        stats['removed_plot'] += 1
                    # elif not keep_second_noisy and 'Noisy' in file:
                    #     # Keep only first noisy image for each reference
                    #     ref_name = file.split('Noisy')[0] + 'Reference.bmp'
                    #     ref_path = os.path.join(root, ref_name)
                    #     if os.path.exists(ref_path):
                    #         existing_noisy = glob.glob(os.path.join(root, f"{file.split('Noisy')[0]}Noisy*.bmp"))
                    #         if len(existing_noisy) > 1 and file_path != existing_noisy[0]:
                    #             should_remove = True
                    #             stats['removed_second_noisy'] += 1
                    
                    if should_remove:
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        stats['space_saved_mb'] += size_mb
                        
                        if move_deleted:
                            # Move to deleted folder
                            new_path = os.path.join(deleted_folder, 
                                                  f"{camera}_{os.path.basename(root)}_{file}")
                            shutil.move(file_path, new_path)
                        else:
                            # Remove file
                            os.remove(file_path)

                if not keep_second_noisy:
                    noisy_images = sorted([f for f in files if f.endswith('Noisy.bmp')])
                        
                    # Keep only the first noisy image if more than one is present
                    if len(noisy_images) > 1:
                        for extra_noisy in noisy_images[1:]:
                            file_path = os.path.join(root, extra_noisy)
                            stats['removed_second_noisy'] += 1
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            stats['space_saved_mb'] += size_mb
                            
                            # Move to the extra noisy folder, preserving structure
                            relative_path = os.path.relpath(root, self.dataset_path)
                            extra_folder_path = os.path.join(self.extra_noisy_path, relative_path)
                            os.makedirs(extra_folder_path, exist_ok=True)
                            shutil.move(file_path, os.path.join(extra_folder_path, extra_noisy))
        return stats

# Example usage
def cleanup_renoir_dataset(dataset_path: str):
    """Run the cleanup process and print statistics"""
    cleaner = RenoirDatasetCleaner(dataset_path)
    
    # Analyze initial space usage
    print("Analyzing initial space usage...")
    initial_stats = cleaner.analyze_space()
    # print(type(initial_stats))
    # print('columns:      ',initial_stats.columns)
    initial_total = initial_stats['size_mb'].sum()
    
    # Print initial statistics
    print("\nInitial space usage by file type:")
    print(initial_stats.groupby('file_type')['size_mb'].agg(['sum', 'count']))
    
    # Run cleanup
    print("\nCleaning up dataset...")
    cleanup_stats = cleaner.cleanup_dataset(keep_second_noisy=False, move_deleted=True)
    
    # Print cleanup statistics
    print("\nCleanup Statistics:")
    print(f"Removed {cleanup_stats['removed_full']} full images")
    print(f"Removed {cleanup_stats['removed_mask']} mask files")
    print(f"Removed {cleanup_stats['removed_plot']} plot files")
    print(f"Removed {cleanup_stats['removed_second_noisy']} second noisy images")
    print(f"Total space saved: {cleanup_stats['space_saved_mb']:.2f} MB")
    print(f"Space saved: {(cleanup_stats['space_saved_mb']/initial_total)*100:.1f}% of original")

    print("\nAnalyzing space usage after clean up...")
    initial_stats = cleaner.analyze_space()
    # print(type(initial_stats))
    # print('columns:      ',initial_stats.columns)
    initial_total = initial_stats['size_mb'].sum()
    
    # Print initial statistics
    print("\nafter clean up space usage by file type:")
    print(initial_stats.groupby('file_type')['size_mb'].agg(['sum', 'count']))



# Initialize the cleaner
renoir_path = 'data/RENOIR'
cleanup_renoir_dataset(renoir_path)



