import os
from .common_utils import *


        
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np


def get_noisy_image_Gaussian(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
        cov_matrix: covariance matrix for multivariate Gaussian noise
    """
    mean = (0, 0,0)
    cov = np.array([[sigma*2, sigma*3, sigma],
               [0, sigma*3, sigma*2],
               [0, 0, sigma]])

    # Generate multivariate Gaussian noise
    noise = np.random.multivariate_normal(mean=mean, cov=cov, size=img_np.shape[1:])

    # Add noise to the image
    noise_reshaped = noise.reshape(img_np.shape)  # Shape should be (3, 512, 512)
    img_noisy_np = np.clip(img_np + noise_reshaped, 0, 1).astype(np.float32)
    
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np

def get_noisy_image_uniform(img_np, intensity):
    """Adds uniform noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        intensity: std of the noise
    """
        # Convert the image to numpy array
    image_np = np.array(image, dtype=np.float32)
    rows, cols, channels = image.shape
    img_noisy_np = np.clip(img_np + np.random.uniform(0.1,0.9, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
