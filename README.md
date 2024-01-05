# MNIST Digit Recognition with CNN and CSV to Image Conversion

This Python project combines two primary scripts, `Training.py` and `Testing.py`, for training and testing a Convolutional Neural Network (CNN) model on the MNIST dataset. Additionally, a supplementary script, `CSV_to_Image.py`, facilitates the conversion of CSV data to a grayscale image.

## Overview

### `Training.py`

- `Training.py` is dedicated to training the CNN model using the MNIST dataset.
- Utilizes PyTorch and torchvision for deep learning functionality.
- Defines a CNN architecture, conducts training epochs, and outputs the trained model parameters.

### `Testing.py`

- `Testing.py` loads the trained CNN model from `Training.py` and applies it to predict the digit in a new grayscale image.
- Utilizes torchvision and PIL for image processing.
- Demonstrates how to preprocess an input image and obtain a digit prediction.

### `CSV_to_Image.py`

- `CSV_to_Image.py` is a supplementary script for converting CSV data to a visual representation.
- Reads CSV data from 'trainData.csv' (adjustable as needed).
- Reshapes the data into a 28x28 array (assuming MNIST image size) and converts it to a grayscale image.
- Saves the resulting image as 'output.png'.

## Instructions for Use:

1. **Training the CNN Model:**
   - Run `Training.py` to train the CNN model on the MNIST dataset.

2. **Testing the Trained Model:**
   - Execute `Testing.py` with a grayscale image path to predict the digit using the trained CNN model.

3. **CSV to Image Conversion:**
   - Optionally, run `CSV_to_Image.py` to convert CSV data to a visual representation.
   - Ensure that 'trainData.csv' is present and adjust the output image filename as needed.

## Dependencies:

- PyTorch
- torchvision
- PIL (Python Imaging Library)
- pandas
- numpy

## Files:

- `Training.py`: Main script for training the CNN model.
- `Testing.py`: Main script for testing the trained model on a new image.
- `CSV_to_Image.py`: Supplementary script for converting CSV data to an image.

## Notes:

- Install the required dependencies using `pip install -r requirements.txt`.
- Adjust file paths, dataset locations, and other parameters based on your project's setup.
- Ensure the presence of the MNIST dataset locally in the 'data' directory for training.
- Images used for testing should be grayscale and resized to 28x28 pixels.
- The CSV conversion script assumes 'trainData.csv' is present and outputs the resulting image as 'output.png'.
- Customize the scripts to suit your specific use case or dataset.
