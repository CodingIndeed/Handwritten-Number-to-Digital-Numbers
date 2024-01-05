# MNIST Digit Recognition with CNN

This project consists of two Python files, `Training.py` and `Testing.py`, for training and testing a Convolutional Neural Network (CNN) model on the MNIST dataset. Additionally, a supplementary script `CSV_to_Image.py` is provided to convert CSV data to image files.

## Training.py

- `Training.py` is responsible for loading the MNIST dataset, defining the CNN model architecture, training the model, and saving the trained model's parameters.

## Testing.py

- `Testing.py` loads the trained model from `Training.py` and uses it to predict the digit in a new grayscale image.

## CSV_to_Image.py

- `CSV_to_Image.py` converts CSV data to a grayscale image. It reads CSV data from 'trainData.csv', reshapes it into a 28x28 array (assuming MNIST image size), and saves the resulting image as 'output.png'.

### Instructions for Use:

1. Run `Training.py` to train the CNN model and save the trained parameters in `mnist_cnn.pth`.
2. Run `Testing.py` with a grayscale image path to predict the digit.
3. Optionally, run `CSV_to_Image.py` to convert CSV data to an image.

### Dependencies:

- PyTorch
- torchvision
- PIL
- pandas
- numpy

### Files:

- `Training.py`: Training script.
- `Testing.py`: Testing script.
- `CSV_to_Image.py`: Script to convert CSV to an image.
- `mnist_cnn.pth`: Trained model parameters.

### Notes:

- Make sure to install the required dependencies before running the scripts.
- The training script assumes that the MNIST dataset is available locally in the 'data' directory.
- The testing script requires a grayscale image for prediction.
- The CSV conversion script assumes 'trainData.csv' is present and outputs the resulting image as 'output.png'.
- Adjust the paths and parameters as needed.
