import os
import pandas as pd
from pathlib import Path

def get_patch_pair_paths(patches_dir):
    """
    Generates a DataFrame with the file paths for clean and noisy patches
    
    Args:
        patches_dir (str or Path): Base directory containing the clean and noisy patches
    
    Returns:
        pandas.DataFrame: with columns 'clean_path' and 'noisy_path'
    """
    patches_dir = Path(patches_dir)
    clean_dir = patches_dir / 'clean'
    noisy_dir = patches_dir / 'noisy'
    
    clean_paths = sorted(list(clean_dir.glob('*.png')))
    noisy_paths = [noisy_dir / clean_path.name for clean_path in clean_paths]
    
    data = {
        'clean_path': [str(p) for p in clean_paths],
        'noisy_path': [str(p) for p in noisy_paths]
    }
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)

# patches_dir = 'data/patches'
# df = get_patch_pair_paths(patches_dir)
# print(df.head())