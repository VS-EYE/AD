#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

# Load a medical image from local drive
# Replace 'path_to_your_image.jpg' with the actual path to your image file
image_path = r"D:\HU\Kaggle\Training\Tr-gl_0099.jpg"  # Update this path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Check if the image was loaded successfully
if image is None:
    raise ValueError("Image not found. Please check the file path.")

# Perform 2D Discrete Wavelet Transform
wavelet = 'coif3'  # You can choose different wavelets (e.g., 'db1', 'sym2', etc.)  #sym5
coeffs = pywt.wavedec2(image, wavelet)

# Extract approximation and detail coefficients
cA, (cH, cV, cD) = coeffs[0], coeffs[1]

# Plot original image and DWT coefficients
plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Approximation Coefficients
plt.subplot(2, 2, 2)
plt.title('Approximation Coefficients (cA)')
plt.imshow(cA, cmap='gray')
plt.axis('off')

# Horizontal Detail Coefficients
plt.subplot(2, 2, 3)
plt.title('Horizontal Detail Coefficients (cH)')
plt.imshow(cH, cmap='gray')
plt.axis('off')

# Vertical Detail Coefficients
plt.subplot(2, 2, 4)
plt.title('Vertical Detail Coefficients (cV)')
plt.imshow(cV, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Thresholding the detail coefficients for denoising (optional)
threshold = 0.1  # Set a threshold value
cH_thresholded = pywt.threshold(cH, threshold, mode='soft')
cV_thresholded = pywt.threshold(cV, threshold, mode='soft')
cD_thresholded = pywt.threshold(cD, threshold, mode='soft')

# Reconstruct the image with the thresholded coefficients
coeffs_thresholded = [cA, (cH_thresholded, cV_thresholded, cD_thresholded)]
reconstructed_image = pywt.waverec2(coeffs_thresholded, wavelet)

# Plot the denoised image
plt.figure(figsize=(6, 6))
plt.title('Denoised Image')
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.show()

pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

# Set the path to the folder containing your images
folder_path = r"D:\Papers\Self\Curtin\Medical Group\Paer -1 (ONE to Multi)\D1"  # Update this path
output_folder = r"D:\Papers\Self\Curtin\Medical Group\Paer -1 (ONE to Multi)\D3"  # Update this output path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process and save images
def process_image(image_path, output_path, wavelet='coif3', threshold=0.1):
    # Load image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image not found at {image_path}. Please check the file path.")

    original_height, original_width = image.shape  # Store original dimensions

    # Perform 2D Discrete Wavelet Transform
    coeffs = pywt.wavedec2(image, wavelet)

    
    # Extract approximation and detail coefficients
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]

    # Thresholding the detail coefficients for denoising (optional)
    #cH_thresholded = pywt.threshold(cH, threshold, mode='soft')
    #cV_thresholded = pywt.threshold(cV, threshold, mode='soft')
    #cD_thresholded = pywt.threshold(cD, threshold, mode='soft')

    # Reconstruct the image with the thresholded coefficients
    #coeffs_thresholded = [cA, (cH_thresholded, cV_thresholded, cD_thresholded)]
    reconstructed_image = pywt.waverec2(coeffs, wavelet)

    # Ensure reconstructed image has the correct shape
    reconstructed_image = np.clip(reconstructed_image, 0, 255)  # Clip pixel values to valid range
    reconstructed_image = reconstructed_image.astype(np.uint8)

    # Crop or pad to match the original size if necessary (this is a safeguard)
    if reconstructed_image.shape != (original_height, original_width):
        reconstructed_image = cv2.resize(reconstructed_image, (original_width, original_height))

    # Save the reconstructed image with cmap='gray' using Matplotlib
    plt.imsave(output_path, reconstructed_image, cmap='gray')

    print(f"Processed and saved: {output_path}")

# Loop through all the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # You can add more formats if necessary
        image_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, f"reconstructed_{filename}")

        # Process each image and save the output with the original dimensions and cmap='gray'
        process_image(image_path, output_path)

print("All images have been processed and saved.")

import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt

def fuseCoeff(cooef1, cooef2, method):
    if method == 'mean':
        cooef = (cooef1 + cooef2) / 2
    elif method == 'min':
        cooef = np.minimum(cooef1, cooef2)
    elif method == 'max':
        cooef = np.maximum(cooef1, cooef2)
    else:
        cooef = []
    return cooef

# Function to process each channel separately
def process_channel(channel1, channel2, wavelet, method):
    cooef1 = pywt.wavedec2(channel1, wavelet)
    cooef2 = pywt.wavedec2(channel2, wavelet)
    fusedCooef = []

    for i in range(len(cooef1)):
        if i == 0:
            fusedCooef.append(fuseCoeff(cooef1[0], cooef2[0], method))
        else:
            # Fuse each set of coefficients
            c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], method)
            c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], method)
            c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], method)
            fusedCooef.append((c1, c2, c3))

    # Reconstruct the image from the fused coefficients
    return pywt.waverec2(fusedCooef, wavelet)

# Params
FUSION_METHOD = 'mean'  # Can be 'min' || 'max || other
wavelet = 'coif3'

# Read the two images
I1 = cv2.imread(r"C:\Users\DS\Desktop\P2.png")
I2 = cv2.imread(r"C:\Users\DS\Desktop\P3.png")

# Assuming both images are the same size
# Convert images from BGR to RGB
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

# Decompose and fuse each channel
fusedR = process_channel(I1[:,:,0], I2[:,:,0], wavelet, FUSION_METHOD)
fusedG = process_channel(I1[:,:,1], I2[:,:,1], wavelet, FUSION_METHOD)
fusedB = process_channel(I1[:,:,2], I2[:,:,2], wavelet, FUSION_METHOD)

# Stack the channels back into an RGB image
fusedImage = cv2.merge([fusedR, fusedG, fusedB])
fusedImage = np.clip(fusedImage, 0, 255)  # Ensure the pixel values are valid
fusedImage = fusedImage.astype(np.uint8)  # Convert to uint8

# Show the fused image
plt.figure(figsize=(6, 6))
plt.title('Fused RGB Image')
plt.imshow(fusedImage)
plt.axis('off')
plt.show()

# Save the fused image
plt.imsave(r"C:\Users\DS\Desktop\fused.png", fusedImage)

import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt

def fuseCoeff(cooef1, cooef2, method):
    if method == 'mean':
        cooef = (cooef1 + cooef2) / 2
    elif method == 'min':
        cooef = np.minimum(cooef1, cooef2)
    elif method == 'max':
        cooef = np.maximum(cooef1, cooef2)
    else:
        cooef = []
    return cooef

# Function to process each channel separately
def process_channel(channel1, channel2, wavelet, method):
    cooef1 = pywt.wavedec2(channel1, wavelet)
    cooef2 = pywt.wavedec2(channel2, wavelet)
    fusedCooef = []

    for i in range(len(cooef1)):
        if i == 0:
            fusedCooef.append(fuseCoeff(cooef1[0], cooef2[0], method))
        else:
            # Fuse each set of coefficients
            c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], method)
            c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], method)
            c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], method)
            fusedCooef.append((c1, c2, c3))

    # Reconstruct the image from the fused coefficients
    return pywt.waverec2(fusedCooef, wavelet)

# Params
FUSION_METHOD = 'mean'  # Can be 'min' || 'max || other
wavelet = 'coif3'

# Read the two images
I1 = cv2.imread(r"D:\Others\enhancedCrack.png")
I2 = cv2.imread(r"D:\Others\enhancedCRACKGA.png")

# Assuming both images are the same size
# Convert images from BGR to RGB
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

# Decompose and fuse each channel
fusedR = process_channel(I1[:,:,0], I2[:,:,0], wavelet, FUSION_METHOD)
fusedG = process_channel(I1[:,:,1], I2[:,:,1], wavelet, FUSION_METHOD)
fusedB = process_channel(I1[:,:,2], I2[:,:,2], wavelet, FUSION_METHOD)

# Stack the channels back into an RGB image
fusedImage = cv2.merge([fusedR, fusedG, fusedB])
fusedImage = np.clip(fusedImage, 0, 255)  # Ensure the pixel values are valid
fusedImage = fusedImage.astype(np.uint8)  # Convert to uint8

# Show the fused image
plt.figure(figsize=(6, 6))
plt.title('Fused RGB Image')
plt.imshow(fusedImage)
plt.axis('off')
plt.show()

# Save the fused image
plt.imsave(r'D:\Others\fused.png', fusedImage)


# In[10]:


import pywt
import cv2
import numpy as np
import os
from PIL import Image

# Function to fuse coefficients according to the specified method
def fuseCoeff(cooef1, cooef2, method):
    if method == 'mean':
        return (cooef1 + cooef2) / 2
    elif method == 'min':
        return np.minimum(cooef1, cooef2)
    elif method == 'max':
        return np.maximum(cooef1, cooef2)
    else:
        return []

# Parameters
#FUSION_METHOD = 'mean'  # Can be 'min', 'max', or 'mean'

# Define the input and output directories
input_dir_1 = r"D:\Papers\Self\Curtin\Medical Group\Paer -1 (ONE to Multi)\D1"  # Update with your image folder path
input_dir_2 = r"D:\Papers\Self\Curtin\Medical Group\Paer -1 (ONE to Multi)\D2"  # Update with your image folder path
output_dir  = r"D:\Papers\Self\Curtin\Medical Group\Paer -1 (ONE to Multi)\D3"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get a list of image files in the input directories
image_files_1 = sorted([f for f in os.listdir(input_dir_1) if f.endswith(('.bmp', '.jpg', '.png'))])
image_files_2 = sorted([f for f in os.listdir(input_dir_2) if f.endswith(('.bmp', '.jpg', '.png'))])

# Process each pair of images in the folders
for i in range(len(image_files_1)):
    # Read the two images
    I1 = cv2.imread(os.path.join(input_dir_1, image_files_1[i]), 0)
    I2 = cv2.imread(os.path.join(input_dir_2, image_files_2[i]), 0)

    # Resize I2 to match I1 if necessary
    if I1.shape != I2.shape:
        I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    # Fusion algorithm
    wavelet = 'coif3'
    cooef1 = pywt.wavedec2(I1, wavelet)
    cooef2 = pywt.wavedec2(I2, wavelet)

    # Fuse coefficients
    fusedCooef = []
    for j in range(len(cooef1) - 1):
        if j == 0:
            fusedCooef.append(fuseCoeff(cooef1[0], cooef2[0], FUSION_METHOD))
        else:
            c1 = fuseCoeff(cooef1[j][0], cooef2[j][0], FUSION_METHOD)
            c2 = fuseCoeff(cooef1[j][1], cooef2[j][1], FUSION_METHOD)
            c3 = fuseCoeff(cooef1[j][2], cooef2[j][2], FUSION_METHOD)
            fusedCooef.append((c1, c2, c3))

    # Inverse wavelet transform to get the fused image
    fusedImage = pywt.waverec2(fusedCooef, wavelet)

    # Normalize values to uint8
    fusedImage = np.clip(fusedImage, 0, None)  # Clip to avoid negative values
    fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage), (np.max(fusedImage) - np.min(fusedImage))), 255)
    fusedImage = fusedImage.astype(np.uint8)
    

    # Resize the fused image to match the original image dimensions
    fusedImage = cv2.resize(fusedImage, (I1.shape[1], I1.shape[0]))
    fusedImage_rgb = cv2.cvtColor(fusedImage, cv2.COLOR_BGR2RGB)
    

    # Save the fused image with the original size
    output_filename = f"fused_{i + 1}.png"  # Generate a unique name
    image = Image.fromarray(fusedImage_rgb)
    #cv2.imwrite(os.path.join(output_dir, output_filename), fusedImage, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # Save as PNG with no compression
    output = os.path.join(output_dir, output_filename)
    image.save(output, format="PNG", compress_level=0)
    

print("Processing complete!")


# In[22]:


import cv2
import numpy as np
from PIL import Image

# Function to read a grayscale image and save it as BGR
def convert_grayscale_to_bgr(input_image_path, output_image_path):
    # Read the image in grayscale (0 flag reads the image in grayscale)
    grayscale_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded correctly
    if grayscale_image is None:
        print("Error: Unable to load image.")
        return
    
    # Convert the grayscale image to BGR by duplicating the grayscale value across all three channels
    bgr_image = np.stack([grayscale_image] * 3, axis=-1)
    
    # Check if the resulting image is indeed BGR (3 channels)
    print(f"Converted image shape: {bgr_image.shape}")

    # Save the resulting BGR image using Pillow (PIL)
    bgr_image_pil = Image.fromarray(bgr_image)
    bgr_image_pil.save(output_image_path)
    print(f"Image saved as {output_image_path}")

# Example usage
input_image_path = r"D:\Papers\Self\Curtin\Medical Group\Paer -1 (ONE to Multi)\D3\fused_1.png"  # Change this to the path of your grayscale image
output_image_path = 'output_bgr_image.png'  # Change this to your desired output path

convert_grayscale_to_bgr(input_image_path, output_image_path)


# In[13]:


import pywt
import cv2
import numpy as np
import os

# Function to fuse coefficients according to the specified method
def fuseCoeff(cooef1, cooef2, method):
    if method == 'mean':
        return (cooef1 + cooef2) / 2
    elif method == 'min':
        return np.minimum(cooef1, cooef2)
    elif method == 'max':
        return np.maximum(cooef1, cooef2)
    else:
        return []

# Parameters
FUSION_METHOD = 'mean'  # Can be 'min', 'max', or 'mean'

# Define the input and output directories
input_dir_1 = r"G:\JOB\Research\China Client\GEO\P2-Q1- Bhai\Fused-Images\PP1\CLAHE\Test\Leakage"  # Update with your image folder path
input_dir_2 = r"G:\JOB\Research\China Client\GEO\P2-Q1- Bhai\Fused-Images\PP1\Correction\Test\Leakage"  # Update with your image folder path
output_dir  = r"G:\JOB\Research\China Client\GEO\P2-Q1- Bhai\Fused-Images\PP1\FUSED\Test\Leakage"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get a list of image files in the input directories
image_files_1 = sorted([f for f in os.listdir(input_dir_1) if f.endswith(('.bmp', '.jpg', '.png'))])
image_files_2 = sorted([f for f in os.listdir(input_dir_2) if f.endswith(('.bmp', '.jpg', '.png'))])

# Process each pair of images in the folders
for i in range(len(image_files_1)):
    # Read the two images in color
    I1 = cv2.imread(os.path.join(input_dir_1, image_files_1[i]), 1)
    I2 = cv2.imread(os.path.join(input_dir_2, image_files_2[i]), 1)

    # Resize I2 to match I1 if necessary
    if I1.shape != I2.shape:
        I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

    # Fusion algorithm
    wavelet = 'coif3'
    fusedImage = np.zeros_like(I1)
    for channel in range(3):  # Process each channel: R, G, B
        cooef1 = pywt.wavedec2(I1[:,:,channel], wavelet)
        cooef2 = pywt.wavedec2(I2[:,:,channel], wavelet)

        # Fuse coefficients
        fusedCooef = []
        for j in range(len(cooef1)):
            if j == 0:
                fusedCooef.append(fuseCoeff(cooef1[0], cooef2[0], FUSION_METHOD))
            else:
                c1 = fuseCoeff(cooef1[j][0], cooef2[j][0], FUSION_METHOD)
                c2 = fuseCoeff(cooef1[j][1], cooef2[j][1], FUSION_METHOD)
                c3 = fuseCoeff(cooef1[j][2], cooef2[j][2], FUSION_METHOD)
                fusedCooef.append((c1, c2, c3))

        # Inverse wavelet transform to get the fused image
        fusedImage[:,:,channel] = pywt.waverec2(fusedCooef, wavelet)

    # Normalize values to uint8
    fusedImage = np.clip(fusedImage, 0, 255).astype(np.uint8)

    # Save the fused image with the original size
    output_filename = f"fused_{i + 1}.png"  # Generate a unique name
    cv2.imwrite(os.path.join(output_dir, output_filename), fusedImage, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # Save as PNG with no compression

print("Processing complete!")

