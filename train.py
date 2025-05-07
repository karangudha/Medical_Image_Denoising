import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset import preprocess_data
from src.model import autoencoder
from src.utility import PSNR

from skimage.metrics import structural_similarity
import lpips
import torch

# Enable GPU memory growth (optional but recommended on Colab)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth set.")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

# Ensure directories exist
os.makedirs('output', exist_ok=True)
os.makedirs(os.path.join('output', 'images'), exist_ok=True)

# Paths to data folders
mri_folder = os.path.join('data', 'mri')
ct_folder = os.path.join('data', 'ct')
xray_folder = os.path.join('data', 'xray')
output_folder = os.path.join('output', 'images')

# Load and preprocess all datasets
try:
    all_clean, all_noisy = preprocess_data(mri_folder, ct_folder, xray_folder)

    # Shuffle the data
    indices = np.arange(all_clean.shape[0])
    np.random.shuffle(indices)
    all_clean = all_clean[indices]
    all_noisy = all_noisy[indices]

    print("All data loaded and shuffled successfully.")
except Exception as e:
    print(f"Error in loading datasets: {e}")
    raise

# Ensure the data has the required shape for the model
if len(all_clean.shape) == 3:
    all_clean = all_clean[..., np.newaxis]
    all_noisy = all_noisy[..., np.newaxis]

# Split data for training and testing
split = int(0.8 * len(all_clean))
x_train_data = all_noisy[:split]
y_train_data = all_clean[:split]
x_test_data = all_noisy[split:]
y_test_data = all_clean[split:]

print(f"Training data shape: {x_train_data.shape}")
print(f"Testing data shape: {x_test_data.shape}")

# Create and compile the autoencoder model
try:
    model = autoencoder()
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    print("Model compiled successfully.")
except Exception as e:
    print(f"Error in model creation/compilation: {e}")
    raise

# Train the model
try:
    model.fit(
        x_train_data, y_train_data,
        epochs=51, batch_size=8,
        validation_data=(x_test_data, y_test_data)
    )
    print("Model training completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Save the model
try:
    model_path = os.path.join('output', 'autoencoder_combined.keras')
    model.save(model_path)
    print(f"Model saved at {model_path}")
except Exception as e:
    print(f"Error saving the model: {e}")
    raise

# Predict on test samples
try:
    predictions = model.predict(x_test_data)
    print("Prediction completed.")
except Exception as e:
    print(f"Error during prediction: {e}")
    raise

# Plot and Save Sample Predictions
try:
    for i, (original, predicted) in enumerate(zip(x_test_data[:5], predictions[:5])):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original.squeeze(), cmap='gray')
        axes[0].set_title('Original (Noisy)')
        axes[0].axis('off')

        axes[1].imshow(predicted.squeeze(), cmap='gray')
        axes[1].set_title('Denoised')
        axes[1].axis('off')

        out_path = os.path.join(output_folder, f'prediction_{i+1}.jpg')
        plt.savefig(out_path, format='jpg')
        plt.close(fig)
        print(f"Saved: {out_path}")

        pred_img = (predicted.squeeze() * 255).astype(np.uint8)
        Image.fromarray(pred_img).save(os.path.join(output_folder, f'denoised_{i+1}.jpg'))
except Exception as e:
    print(f"Error during plotting predictions: {e}")

# Evaluate using multiple metrics and save the graph
try:
    print("\n--- Evaluating Metrics ---")

    psnr_scores = []
    ssim_scores = []
    mae_scores = []
    lpips_scores = []

    loss_fn_alex = lpips.LPIPS(net='alex')

    for clean, pred in zip(y_test_data, predictions):
        clean_img = clean.squeeze()
        pred_img = pred.squeeze()

        psnr_val = PSNR(clean_img, pred_img)
        ssim_val = structural_similarity(clean_img, pred_img, data_range=1.0)
        mae_val = tf.reduce_mean(tf.abs(clean_img - pred_img)).numpy()

        clean_tensor = torch.tensor(np.stack([clean_img]*3, axis=0)).unsqueeze(0).float()
        pred_tensor = torch.tensor(np.stack([pred_img]*3, axis=0)).unsqueeze(0).float()

        lpips_val = loss_fn_alex(clean_tensor, pred_tensor).item()

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        mae_scores.append(mae_val)
        lpips_scores.append(lpips_val)

    print(f"\nMetrics over {len(predictions)} test images:")
    print(f"PSNR  (Higher Better) : {np.mean(psnr_scores):.2f} dB")
    print(f"SSIM  (Higher Better) : {np.mean(ssim_scores):.4f}")
    print(f"MAE   (Lower Better)  : {np.mean(mae_scores):.4f}")
    print(f"LPIPS (Lower Better)  : {np.mean(lpips_scores):.4f}")

    history = model.fit(
    x_train_data, y_train_data,
    epochs=51, batch_size=8,
    validation_data=(x_test_data, y_test_data)
    )

    # Save training/validation loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'loss_graph.png'))
    plt.close()
    print("Saved: output/loss_graph.png")


except Exception as e:
    print(f"Error during metric evaluation: {e}")
