import os
import gdown  # Install using: pip install gdown
import zipfile

# Google Drive shareable links (replace with your actual links)
clean_url = "https://drive.google.com/uc?id=1Yzo_2TeJuivg5mTKZf6iNzFrgyqP4nXP"
# https://drive.google.com/file/d/1Yzo_2TeJuivg5mTKZf6iNzFrgyqP4nXP/view?usp=drive_link
noisy_url = "https://drive.google.com/uc?id=14XqiNwm-ejOdfMZ95u-RjBYJ43qvNDDK"
#https://drive.google.com/file/d/14XqiNwm-ejOdfMZ95u-RjBYJ43qvNDDK/view?usp=drive_link



# Destination directory
data_dir = "data/patches"

# Create necessary directory
os.makedirs(data_dir, exist_ok=True)

# Download clean.zip
clean_zip_path = os.path.join(data_dir, "clean.zip")
print("Downloading clean.zip...")
gdown.download(clean_url, clean_zip_path, quiet=False)

# Unzip clean.zip directly into data/patches/
print("Extracting clean.zip...")
with zipfile.ZipFile(clean_zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

os.remove(clean_zip_path)

# Download noisy.zip
noisy_zip_path = os.path.join(data_dir, "noisy.zip")
print("Downloading noisy.zip...")
gdown.download(noisy_url, noisy_zip_path, quiet=False)


# Unzip noisy.zip directly into data/patches/
print("Extracting noisy.zip...")
with zipfile.ZipFile(noisy_zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)


os.remove(noisy_zip_path)

print("Data downloaded, extracted, and ready for use.")
