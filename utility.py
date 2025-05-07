import numpy as np 
import matplotlib.pyplot as plt
import cv2

def plot_predictions(y_true, y_pred, x_data, image_shape=(512,512)):
    """Plots original, true (clean), predicted, and median-blurred images side by side."""
    f, ax = plt.subplots(4, 5)
    f.set_size_inches(12, 9)
    for i in range(5):
        ax[0][i].imshow(np.reshape(x_data[i], image_shape), cmap='gray')
        ax[0][i].set_title('Noisy')
        ax[0][i].axis('off')

        ax[1][i].imshow(np.reshape(y_true[i], image_shape), cmap='gray')
        ax[1][i].set_title('Ground Truth')
        ax[1][i].axis('off')

        ax[2][i].imshow(np.reshape(y_pred[i], image_shape), cmap='gray')
        ax[2][i].set_title('Denoised (Pred)')
        ax[2][i].axis('off')

        blurred = cv2.medianBlur(np.reshape(x_data[i], image_shape).astype(np.uint8), 5)
        ax[3][i].imshow(blurred, cmap='gray')
        ax[3][i].set_title('Median Blur')
        ax[3][i].axis('off')

    plt.tight_layout()
    plt.show()

def PSNR(original, denoised):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100.0  # Perfect match
    max_pixel = 1.0 if original.max() <= 1 else 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))
