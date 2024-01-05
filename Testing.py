# Import necessary libraries
from Training import CNN
import torch
import torchvision
from PIL import Image

# Load the model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# Load the grayscale image for testing
input_image_path = "output.png" # image to test
image = Image.open(input_image_path).convert("L") # Convert to grayscale

# Apply transformations to the input image
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (128,))])
input_image_tensor = transform(image)

# Function to predict text from an image
def predict_text(image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        predicted_class = torch.argmax(output).item()
        predicted_text = str(predicted_class)
    return predicted_text

# Get the predicted text from the input image
predicted_text = predict_text(input_image_tensor)
print("Predicted text:", predicted_text)
