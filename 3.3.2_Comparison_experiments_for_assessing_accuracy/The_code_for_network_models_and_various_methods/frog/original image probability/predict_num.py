import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import AlexNet


def sort_func(file_name):
    # Extract the numerical part of the file name
    return int(''.join(filter(str.isdigit, file_name)))


def main():
    # Set the device based on CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Perform initialization operations on the images
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Define the folder path
    folder_path = r"D:\imagenet\frog\yuan tu"

    # Read class_indict from json file
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create the model
    model = AlexNet(num_classes=2).to(device)

    # Load model weights
    weights_path = "AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()

    # Loop through the images in the folder
    file_list = os.listdir(folder_path)
    file_list = sorted(file_list, key=sort_func)  # Sort file names according to the custom sort function
    for file_name in file_list:
        # Construct the file path
        file_path = os.path.join(folder_path, file_name)

        # Open the image
        img = Image.open(file_path)

        # Preprocess the image
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # Make predictions
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            class_probabilities = predict.numpy()
            class_id = torch.argmax(predict).item()
            class_name = class_indict[str(class_id)]

        # print("File: {:10}   Class: {:10}   Probability: {:.3f}".format(file_name, class_name,
        #                                                                   class_probabilities[class_id]))


        print("File: {:10}     class: {:10}   prob: {:.3f}   raw prob: {:.3f}".format(file_name,  class_indict[str(0)], predict[0].numpy(),
                                                                      output[0].numpy()))

if __name__ == '__main__':
    main()