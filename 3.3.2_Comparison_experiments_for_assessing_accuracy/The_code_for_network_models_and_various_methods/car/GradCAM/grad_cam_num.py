import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from model import AlexNet


def sort_func(file_name):
    # Extract the numerical part of the file name
    return int(''.join(filter(str.isdigit, file_name)))


def main():
    # Load your own model trained on your own training dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=2)  # .to(device)

    # Load model weights
    model_weight_path = "./AlexNet.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # Retrieve the structure of the last layer
    target_layers = [model.features[-1]]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # Define the folder paths
    input_folder_path = "D:\pycharmBao\\biaoge\car\one car"
    output_folder_path = "D:\pycharmBao\\biaoge\car\one car grad cam"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Read the file names from the input folder
    file_list = os.listdir(input_folder_path)
    file_list = sorted(file_list, key=sort_func)  # Sort file names according to the custom sort function

    for file_name in file_list:
        # Construct the input and output file paths
        input_file_path = os.path.join(input_folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, file_name)

        # Load the image
        img = Image.open(input_file_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        # Preprocess the image
        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        # Create the GradCAM object
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

        # Specify the target category
        target_category = 0

        # Generate the CAM
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

        # Save the heatmap image
        image = Image.fromarray((visualization).astype(np.uint8))
        image.save(output_file_path)


if __name__ == '__main__':
    main()