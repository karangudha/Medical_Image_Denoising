import os
import numpy as np
from PIL import Image
import tifffile as tiff


def load_images_from_folder(folder, target_size=(512, 512)):
    """
    Loads and preprocesses all images from a folder (supports .png, .jpg, .jpeg, .tif, .tiff).
    If image is smaller than target_size, adds black padding.
    If image is larger, resizes it down to target_size.
    Converts to grayscale, normalizes pixel values to [0, 1].
    Returns: numpy array of shape (N, H, W, 1)
    """
    images = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    if not os.path.isdir(folder):
        print(f"[Error] Folder not found: {folder}")
        return np.array([])

    filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith(image_extensions)])

    for filename in filenames:
        img_path = os.path.join(folder, filename)
        try:
            if filename.lower().endswith(('.tif', '.tiff')):
                img = tiff.imread(img_path)
                if img.ndim == 3:
                    img = img[:, :, 0]  # Convert to 2D if multi-channel
                img = Image.fromarray(img).convert('L')
            else:
                img = Image.open(img_path).convert('L')

            original_width, original_height = img.size

            # Pad if smaller
            if original_width < target_size[0] or original_height < target_size[1]:
                new_img = Image.new('L', target_size, color=0)  # Black padding
                left = (target_size[0] - original_width) // 2
                top = (target_size[1] - original_height) // 2
                new_img.paste(img, (left, top))
                img = new_img
            else:
                # Resize if larger
                img = img.resize(target_size)

            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)  # Shape: (H, W, 1)
            images.append(img)
        except Exception as e:
            print(f"[Warning] Could not load image {img_path}: {e}")

    if not images:
        print(f"[Warning] No valid images found in {folder}.")
        return np.array([])

    return np.array(images)


def add_noise(image, noise_factor=0.3, mean=0.0, sigma=1.5):
    """
    Adds Gaussian noise to a single image.
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss * noise_factor
    return np.clip(noisy_image, 0.0, 1.0)


def preprocess_data(mri_folder, ct_folder, xray_folder):
    """
    Loads, combines, and adds noise to MRI, CT, and X-ray images.
    Returns: tuple of (clean_images, noised_images)
    """
    print("Loading MRI images...")
    mri_images = load_images_from_folder(mri_folder)

    print("Loading CT images...")
    ct_images = load_images_from_folder(ct_folder)

    print("Loading X-ray images...")
    xray_images = load_images_from_folder(xray_folder)

    all_images_list = []
    for name, dataset in zip(["MRI", "CT", "X-ray"], [mri_images, ct_images, xray_images]):
        if dataset.ndim == 4 and dataset.shape[0] > 0:
            print(f"{name} images loaded: {dataset.shape[0]}")
            all_images_list.append(dataset)
        else:
            print(f"[Info] Skipping {name} images due to invalid shape or no data.")

    if not all_images_list:
        raise ValueError("[Fatal] No valid image data found in any folder.")

    all_images = np.concatenate(all_images_list, axis=0)

    print("Adding noise to images...")
    noised_images = np.array([add_noise(img) for img in all_images])

    return all_images, noised_images
