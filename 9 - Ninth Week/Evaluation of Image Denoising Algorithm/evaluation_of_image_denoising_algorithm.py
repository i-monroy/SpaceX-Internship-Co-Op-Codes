"""
Author: Isaac Monroy
Title: Evaluation of Image Denoising Algorithm
Description:
    The objective of this algorithm is to
    denoise a folder of pictures from the BSDS500 
    dataset so that its performance can be evaluated 
    by the metrics Mean Square Error (MSE) and Peak 
    Signal-To-Noise Ratio (PSNR) to demonstrate how 
    accurate the model is in denoising images from 
    common types of noises that affect images, such 
    as Gaussian, Salt and Pepper, Poisson, and Speckle 
    noises.
"""
# Import necessary modules
import numpy as np # Used for numerical operations
import cv2 # Used for image reading, color conversion, and denoising techniques.
import os # Used for interacting with the filesystem to read images from the folder.

# Functions to add noise
def add_gaussian_noise(image):
    """
    Adds Gaussian noise to the input image by generating 
    a Gaussian distribution with the specified mean and 
    variance
    """
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
    noisy = np.clip(image + gauss, 0, 1).astype(np.float32)
    return noisy

def add_salt_and_pepper_noise(image, prob=0.05):
    """
    Adds salt-and-pepper noise to the input image by 
    randomly selecting a proportion of pixels to be 
    set to 0 (pepper) or 1 (salt)
    """
    noisy = np.copy(image)
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 1

    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def add_poisson_noise(image, lam=10):
    """
    Adds Poisson noise to the input image by generating 
    Poisson-distributed random numbers with the specified 
    lambda value
    """
    scaled_image = np.round(image * 255)
    noisy_image = np.zeros_like(scaled_image, dtype=np.float32)

    for i in range(image.shape[2]):
        positive_lam = np.maximum(scaled_image[:, :, i] * lam, 1)
        noisy_image[:, :, i] = np.random.poisson(positive_lam) / float(lam)

    noisy_image = np.clip(noisy_image, 0, 255) / 255
    return noisy_image

def add_speckle_noise(image, var=0.1):
    """
    Adds speckle noise to the input image by generating 
    random Gaussian noise and multiplying it by the input 
    image
    """
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch).reshape(row, col, ch)
    noisy = image + image * gauss * var
    return noisy

# Anscombe transform and its inverse
def anscombe_transform(image):
    """
    Applies the Anscombe transform to the input image, 
    which stabilizes the variance of Poisson noise 
    across the intensity range
    """
    return 2 * np.sqrt(image.astype(np.float32) * 255 + 3 / 8)

def inverse_anscombe_transform(image):
    """
    Applies the inverse Anscombe transform to the input 
    image, converting it back to its original intensity 
    range after denoising
    """
    return ((image / 2)**2 - 3 / 8) / 255

# Poisson and Speckle denoising functions
def remove_poisson_noise(image, h=4):
    """
    Removes Poisson noise from the input image by applying 
    the Anscombe transform, denoising using Non-Local Means, 
    and then applying the inverse Anscombe transform
    """
    transformed_image = anscombe_transform(image)
    transformed_image_8bit = np.clip(transformed_image, 0, 255).astype(np.uint8)
    denoised_transformed_image = cv2.fastNlMeansDenoisingColored(transformed_image_8bit, None, h, h, 7, 21)
    denoised_transformed_image_float = denoised_transformed_image.astype(np.float32)
    denoised_image = inverse_anscombe_transform(denoised_transformed_image_float)
    denoised_image = np.clip(denoised_image, 0, 255).astype(image.dtype)
    return denoised_image

def remove_speckle_noise(image, h=6):
    """
    Removes speckle noise from the input image by 
    converting it to 8-bit, denoising using Non-Local 
    Means, and converting it back to float32
    """
    image_8bit = np.clip(image * 255, 0, 255).astype(np.uint8)
    denoised_image = cv2.fastNlMeansDenoisingColored(image_8bit, None, h, h, 7, 21)
    denoised_image_float = denoised_image.astype(np.float32) / 255
    return denoised_image_float

# Metrics: Mean squared error (MSE) and Peak signal-to-noise ratio (PSNR)
def mse(img1, img2):
    """
    Calculates the mean squared error (MSE) between two 
    images, which measures the average squared differences 
    between corresponding pixel values
    """
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    """
    Calculates the peak signal-to-noise ratio (PSNR) 
    between two images, which measures the quality of 
    an image relative to the maximum possible pixel value
    """
    mse_value = mse(img1, img2)
    if mse_value == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

psnr_values = []
mse_values = []

i = 0
folder_dir = "./train/clean"

# Process first 50 images in the folder
for images in os.listdir(folder_dir):
    if images.endswith(".jpg"):
        img_path = os.path.join(folder_dir, images)
        img = cv2.imread(img_path)
        
        # Read and preprocess the image
        color_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        clean_img = color_img.astype(np.float32) / 255.0

        # Add different types of noise to the clean image
        noisy_image = add_gaussian_noise(clean_img)
        noisy_image = add_salt_and_pepper_noise(noisy_image)
        noisy_image = add_poisson_noise(noisy_image)
        noisy_image = add_speckle_noise(noisy_image)
        
        # Apply denoising techniques to the noisy image
        denoised_image = remove_poisson_noise(noisy_image)
        denoised_image = cv2.GaussianBlur((denoised_image * 255).astype(np.uint8), (3, 3), 0).astype(np.float32) / 255
        denoised_image = cv2.medianBlur((denoised_image * 255).astype(np.uint8), 3).astype(np.float32) / 255
        denoised_image = remove_speckle_noise(denoised_image, h=9)
        
        # Calculate and store PSNR and MSE values for the clean and denoised images
        psnr_values.append(psnr(clean_img, denoised_image))
        mse_values.append(mse(clean_img, denoised_image))
        
        i += 1
        if i >= 50:
            break

# Calculate and print average MSE and PSNR values
psnr_np = np.array(psnr_values)
mse_np = np.array(mse_values)

psnr_avg = np.mean(psnr_np)
mse_avg = np.mean(mse_np)

print(f"MSE: {mse_avg}")
print(f"PSNR: {psnr_avg} dB")

