# Import necessary libraries
import pandas as pd
import numpy as np
from PIL import Image

# Read CSV data from file
df = pd.read_csv('trainData.csv', header=None) # The name of the CSV file

# Convert DataFrame to numpy array
data = df.values.flatten()

## Reshape data into a 28x28 array (assuming MNIST image size)
image_data = data.reshape((28, 28))

# Convert the data type to uint8 (8-bit unsigned integer)
image_data = image_data.astype(np.uint8)

# Create an image from the array using PIL
image = Image.fromarray(image_data)

# Save the image
image.save('output.png')
