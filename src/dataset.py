import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder, target_size=(128, 128)):
    """
    Loads and preprocesses all images from a folder, including .tif/.tiff files.
    Converts them to grayscale, resizes to 128x128, and normalizes to [0, 1].
    """
    images = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith(image_extensions)])

    for filename in filenames:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
            img = img.resize(target_size)
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
    return np.array(images)

def add_noise(image, noise_factor=0.05):
    """
    Adds Gaussian noise to a single image.
    """
    row, col, ch = image.shape
    mean = 0
    sigma = 1
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss * noise_factor
    return np.clip(noisy_image, 0, 1)

def preprocess_data(mri_folder, ct_folder, xray_folder):
    """
    Loads, combines, and adds noise to MRI, CT, and X-ray images.
    Returns clean and noisy versions.
    """
    print("Loading MRI images...")
    mri_images = load_images_from_folder(mri_folder)

    print("Loading CT images...")
    ct_images = load_images_from_folder(ct_folder)

    print("Loading X-ray images...")
    xray_images = load_images_from_folder(xray_folder)

    # Combine datasets
    all_images = np.concatenate([mri_images, ct_images, xray_images], axis=0)

    # Add noise to all images
    noised_images = np.array([add_noise(img) for img in all_images])

    return all_images, noised_images
