import torch
import torch.nn as nn
import torch.nn.functional as F
from model import AlexNet
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Define the Grad-CAM++ class
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_grad = None
        self.feature_map = None
        self.gradient = None
        self.hook()

    def hook(self):
        def feature_hook(module, input, output):
            self.feature_map = output.detach()

        def gradient_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        target_layer = self.target_layer
        assert target_layer is not None, 'Please provide a valid target layer'

        target_layer.register_forward_hook(feature_hook)
        target_layer.register_backward_hook(gradient_hook)

    def __call__(self, input_image, target_class=None):
        self.model.zero_grad()

        # Forward propagation
        input_image.requires_grad_()
        output = self.model(input_image)
        if target_class is None:
            target_class = torch.argmax(output)

        # Backward propagation, compute class-specific feature gradients
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM++ weights
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        activations = F.relu(self.feature_map)
        grad_cam = torch.mean(weights * activations, dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)

        # Multiply the Grad-CAM++ results with the original image and normalize the result
        grad_cam = nn.functional.interpolate(grad_cam, size=(input_image.size(2), input_image.size(3)), mode='bilinear',
                                             align_corners=False)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)

        return grad_cam.squeeze().cpu().numpy()

# Load the AlexNet model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=2)  # .to(device)

# Load model weights
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# Define a sorting function
def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

# Folder path
input_folder = "D:\pycharmBao\\biaoge\\frog\\one frog"
output_folder = "D:\pycharmBao\\biaoge\\frog\one frog grad cam++"

# Retrieve the names of image files in the folder and sort them based on the numeric part
image_files = sorted(os.listdir(input_folder), key=sort_func)

# Create an output folder
os.makedirs(output_folder, exist_ok=True)

# Iterate through image files
for file_name in image_files:
    image_path = os.path.join(input_folder, file_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))
    image = image / 255.0
    image = torch.from_numpy(image).float().unsqueeze(0)

    # Create a Grad-CAM++ object and visualize the results
    grad_cam_plus_plus = GradCAMPlusPlus(model, model.features[-1])
    grad_cam = grad_cam_plus_plus(image, target_class=0)  # Set the class label to 0

    # Visualize the results
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0
    result = heatmap + (image.squeeze().permute(1, 2, 0)).detach().numpy()
    result = result / np.max(result)

    # Construct the output path
    output_path = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}.png')
    cv2.imwrite(output_path, result*255)

    # Display the saved image
    saved_image = cv2.imread(output_path)
    saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)

    plt.imshow(saved_image)
    plt.axis('off')
    plt.show()
