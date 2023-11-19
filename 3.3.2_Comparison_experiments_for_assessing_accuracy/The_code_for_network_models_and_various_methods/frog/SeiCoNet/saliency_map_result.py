from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def sort_func(file_name):
    # Extract the numeric part from the file name
    return int(''.join(filter(str.isdigit, file_name)))

def Generate_preview_image(matrix_R, matrix_G, matrix_B, file_path):
    # Convert the matrix to uint8 data type
    matrix_R = matrix_R.astype(np.uint8)
    matrix_G = matrix_G.astype(np.uint8)
    matrix_B = matrix_B.astype(np.uint8)

    # Create an RGB image
    rgb_array = np.dstack((matrix_R, matrix_G, matrix_B))

    # Convert the matrix back to its original data type.
    rgb_array = rgb_array.astype(matrix_R.dtype)

    # Create a PIL (Python Imaging Library) image object
    image = Image.fromarray(rgb_array)

    # Save the image in PNG format
    file_path_with_extension = file_path + ".png"
    image.save(file_path_with_extension)

# Perform ReLU operation on the matrix
def ReLu_matrix(matrix):
    matrix_temp = np.zeros((224,224))
    for i in range(224):
        for j in range(224):
            if matrix[i][j] > 0:
                matrix_temp[i][j] = matrix[i][j]
    return  matrix_temp

# Min-max normalization
def Max_Min_normalization(test_data):
    return test_data.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))

# Source folder and destination folder paths
source_folder = r"D:\pycharmBao\\biaoge\\frog\\one frog"
target_folder = r"D:\pycharmBao\biaoge\frog\one frog seiconet"

# Retrieve the list of files in the source folder and sort them using a custom sorting function
file_list = os.listdir(source_folder)
file_list = sorted(file_list, key=sort_func)

# Read each file and generate a heatmap, then save it to the destination folder
for file_name in file_list:
    # Construct the complete file path
    file_path = os.path.join(source_folder, file_name)

    # Open the image file and resize it to the specified dimensions
    image = Image.open(file_path)
    image = image.resize((224, 224))

    # Convert to a NumPy array and extract RGB channel data
    image_array = np.array(image)
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # Normalize the RGB channel data
    normalized_red_channel = red_channel / 255.0
    normalized_green_channel = green_channel / 255.0
    normalized_blue_channel = blue_channel / 255.0

    # Read the sensitivity matrix file and process the image data by channel
    sensitivity_value_R = np.loadtxt('sensitivity_value_matrix_R.txt')
    sensitivity_value_G = np.loadtxt('sensitivity_value_matrix_G.txt')
    sensitivity_value_B = np.loadtxt('sensitivity_value_matrix_B.txt')

    sensitivity_level_R = (ReLu_matrix(sensitivity_value_R)) * normalized_red_channel
    sensitivity_level_G = (ReLu_matrix(sensitivity_value_G)) * normalized_green_channel
    sensitivity_level_B = (ReLu_matrix(sensitivity_value_B)) * normalized_blue_channel

    # MinMax normalize the sensitive features
    sensitivity_levels = pd.DataFrame({
        'R': Max_Min_normalization(pd.DataFrame(sensitivity_level_R)).values.flatten(),
        'G': Max_Min_normalization(pd.DataFrame(sensitivity_level_G)).values.flatten(),
        'B': Max_Min_normalization(pd.DataFrame(sensitivity_level_B)).values.flatten()
    })

    # Select sensitive features to generate a heatmap
    heatmap_matrix = sensitivity_levels.max(axis=1).values.reshape(224, 224)

    # Draw the heatmap and save it to the destination folder
    heatmap = cv2.normalize(heatmap_matrix, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    target_file_path = os.path.join(target_folder, file_name)
    cv2.imwrite(target_file_path,  heatmap_color)