import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image
from model import AlexNet

# Define the Score-CAM class
class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = self.model._modules[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, input):
        return self.model(input)

    def get_activation(self):
        return self.activation

    def get_gradients(self):
        return self.gradients

    def generate_cam(self, input, class_idx):
        self.model.zero_grad()
        output = self.forward(input)
        one_hot = torch.zeros(output.size()).to(input.device)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)
        weights = torch.mean(self.get_gradients(), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.get_activation(), dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam

# Load the AlexNet model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=2)  # 请根据实际情况修改类的名称
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# Define a Score-CAM object
score_cam = ScoreCAM(model, target_layer='features')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a sorting function
def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

# Read all images in a folder for processing
input_folder = r"D:\pycharmBao\\biaoge\cat\\two cat"
output_folder = r"D:\pycharmBao\biaoge\cat\two cat ScoreCAM"

# Ensure the output folder exists, and create it if it doesn't
os.makedirs(output_folder, exist_ok=True)

# Retrieve the paths of all image files and sort them in ascending order
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]
image_files.sort(key=lambda x: sort_func(os.path.basename(x)))

for image_path in image_files:
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    image = transform(pil_image)
    image = image.unsqueeze(0)

    # Manually set the preferred category
    target_class = 0


    cam = score_cam.generate_cam(image, target_class)
    cam = cam.squeeze().cpu().numpy()
    image = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Resize the heatmap to match the size of the image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    cam = np.float32(heatmap) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    # Retrieve the file name of the input image (excluding the path)）
    image_name = os.path.basename(image_path)

    # Construct the output file path
    output_path = os.path.join(output_folder, image_name)

    # Save the visualized image
    cv2.imwrite(output_path, cam)
