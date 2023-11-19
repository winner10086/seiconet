import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from new_model import AlexNet


def main():
    # Set the device based on CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Gerform initialization operations on the images
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0, 0,0), (1, 1, 1))])

    # Symbolic execution
    folder_path = '../../images/picture_G'
    assert os.path.exists(folder_path), "Folder '{}' does not exist.".format(folder_path)

    file_list = os.listdir(folder_path)

    # Custom sort function to sort file names by numerical value
    def sort_func(file_name):
        # Extract the numerical part of the file name
        return int(''.join(filter(str.isdigit, file_name)))

    # Sort the file names according to the custom sort function
    file_list = sorted(file_list, key=sort_func)

    # Create two empty lists to store the raw probability values for each category
    class_probs = [[] for _ in range(4)]

    # Read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "File '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = AlexNet(num_classes=4).to(device)

    # Load model weights
    weights_path = './modified_model_deReLu_modweight.pth'
    assert os.path.exists(weights_path), "File '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    output_file_path = '../sensitivity_value/matrix_G.txt'
    with open(output_file_path, "w") as output_file:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            assert os.path.isfile(file_path), "File '{}' does not exist.".format(file_path)

            img = Image.open(file_path)
            img_tensor = data_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                raw_prob = torch.squeeze(output).cpu().numpy()

            for i, prob in enumerate(raw_prob):
                class_probs[i].append(prob.tolist())
        # Bias
        cha = [-9213170,-578921.062,0,0]

        for i, class_prob in enumerate(class_probs):
            class_name = class_indict[str(i)]
            output_file.write("Class: {}, Raw Probabilities: {}\n".format(class_name, class_prob))
            print("{}={}\n".format(class_name, [x - cha[i] for x in class_prob]))

        # Sensitivity_value_matrix
        matrix = [x - cha[0] for x in class_probs[0]]

        flag = 0
        matrix_R= [[0.0] * 224 for _ in range(224)]
        for hang in range(224):
            for lie in range(224):
                if matrix[flag] > 0:
                    matrix_R[hang][lie] = matrix[flag]
                if matrix[flag] < 0:
                    matrix_R[hang][lie] = matrix[flag]
                flag = flag + 1
        print(matrix_R)

        matrix_np = np.array(matrix_R)

        # Save sensitivity_value_matrix
        np.savetxt(r'../sensitivity_value_matrix_G.txt', matrix_np, fmt='%.6f')


if __name__ == '__main__':
    main()