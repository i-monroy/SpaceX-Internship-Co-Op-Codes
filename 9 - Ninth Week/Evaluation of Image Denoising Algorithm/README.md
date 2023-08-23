# Context Bandit MDP Algorithm

## Author
Isaac Monroy

## Project Description
The algorithm denoises a folder of pictures from the BSDS500 dataset using various noise models like Gaussian, Salt and Pepper, Poisson, and Speckle noises. Its performance is evaluated using the metrics Mean Square Error (MSE) and Peak Signal-To-Noise Ratio (PSNR).

## Libraries Used
- **NumPy**: Used for numerical operations and array manipulations.
- **OpenCV (cv2)**: Used for image reading, color conversion, and denoising techniques.
- **os**: Used for interacting with the filesystem to read images from the folder.

## How to Run
1. Ensure that the required libraries (NumPy and OpenCV) are installed.
2. Place the images you want to denoise in the specified folder (e.g., ./train/clean).
3. Run the script and the code will process the images, apply different noise types, denoise them, and calculate the PSNR and MSE metrics.

## Input and Output
- **Input**: Images from the specified folder, assumed to be in the JPEG format.
- **Output**: The average Mean Square Error (MSE) and Peak Signal-To-Noise Ratio (PSNR) of the denoised images relative to the original clean images, printed in the console.
