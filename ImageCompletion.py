import os
import cv2
import numpy as np
from scipy.signal import fftconvolve
import tqdm
import matplotlib.pyplot as plt

def load_images(folder):
    """Load images from the specified folder."""
    inputs, masks, patches = [], [], []
    for i in range(1, 5):
        input_img = cv2.imread(os.path.join(folder, f"input{i}.jpg"), cv2.IMREAD_COLOR)
        mask_img = cv2.imread(os.path.join(folder, f"input{i}_mask.jpg"), cv2.IMREAD_GRAYSCALE)
        patch_img = cv2.imread(os.path.join(folder, f"input{i}_patch.jpg"), cv2.IMREAD_COLOR)

        inputs.append(input_img)
        masks.append(mask_img)
        patches.append(patch_img)

    return inputs, masks, patches

def compute_error_naive(region, patch):
    """Compute the L2 error naively."""
    return np.sum((region - patch) ** 2)

def compute_error_fft(region, patch):
    """Compute the L2 error using FFT for faster computation."""
    return np.sum(
        fftconvolve(region, np.flip(np.flip(patch, 0), 1), mode='valid') ** 2
    )

# Load images
folder = 'data/completion'
inputs, masks, patches = load_images(folder)

# Define parameters
k = 20  # Pixel range for BFS or FFT
use_fft = True  # Use FFT for faster computation

def find_best_patch(input_img, mask_img, patch_img, k=10, use_fft=True):
    """Find the best matching region in the patch image to the masked area in the input image."""
    # Invert the mask: white areas represent the region to patch
    inv_mask = cv2.bitwise_not(mask_img)
    inv_mask = cv2.dilate(inv_mask, np.ones((k, k), np.uint8))
    
    # Initialize variables to keep track of the best match
    min_error = float('inf')
    best_position = (0, 0)
    
    # Slide the patch image over the masked area and compute the sum of squared differences
    for y in range(input_img.shape[0] - patch_img.shape[0]):
        for x in range(input_img.shape[1] - patch_img.shape[1]):
            if use_fft:
                error = compute_error_fft(cv2.bitwise_and(patch_img, patch_img, mask=inv_mask), input_img[y:y + patch_img.shape[0], x:x + patch_img.shape[1]])
            else:
                error = compute_error_naive(cv2.bitwise_and(patch_img, patch_img, mask=inv_mask), input_img[y:y + patch_img.shape[0], x:x + patch_img.shape[1]])

            if error < min_error:
                min_error = error
                best_position = (x, y)
    return best_position

def apply_best_patch(input_img, mask_img, patch_img):
    """Apply the best patch to the input image."""
    best_position = find_best_patch(input_img, mask_img, patch_img)
    patch_x, patch_y = best_position
    h, w = mask_img.shape

    # Extract the best matching patch from the patch image
    best_patch = patch_img[patch_y:patch_y+h, patch_x:patch_x+w]

    # Apply the mask to the input image, blacking out the area to be patched
    input_img_masked = cv2.bitwise_and(input_img, input_img, mask=mask_img)

    # Invert the mask to apply the best patch
    inv_mask = cv2.bitwise_not(mask_img)

    # Combine the input image with the masked area blacked out and the best patch
    return cv2.bitwise_or(input_img_masked, cv2.bitwise_and(best_patch, best_patch, mask=inv_mask))


# Apply the patch with mask to each image and display the results
plt.figure(figsize=(16, 8))
for i, (input_img, mask_img, patch_img) in enumerate(zip(inputs, masks, patches)):
    # Apply the patch considering the mask
    result_img = apply_best_patch(input_img, mask_img, patch_img)

    # Plotting
    plt.subplot(len(inputs), 4, i * 4 + 1)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Input {i+1}")
    plt.axis('off')

    plt.subplot(len(inputs), 4, i * 4 + 2)
    plt.imshow(mask_img, cmap='gray')
    plt.title(f"Mask {i+1}")
    plt.axis('off')

    plt.subplot(len(inputs), 4, i * 4 + 3)
    plt.imshow(cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Patch {i+1}")
    plt.axis('off')

    plt.subplot(len(inputs), 4, i * 4 + 4)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Result {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()