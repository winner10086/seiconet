from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

#########################################################################################################

# Generate an analytical image based on the original image and the architectural characteristics of AlexNet
def Generate_preview_image(matrix_R, matrix_G, matrix_B, file_path):
    # Convert the matrix to uint8 data type
    matrix_R = matrix_R.astype(np.uint8)
    matrix_G = matrix_G.astype(np.uint8)
    matrix_B = matrix_B.astype(np.uint8)

    # Create an RGB image
    rgb_array = np.dstack((matrix_R, matrix_G, matrix_B))

    # Convert the matrix back to its original data type
    rgb_array = rgb_array.astype(matrix_R.dtype)

    # Create a PIL image object
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

#########################################################################################################



#########################################################################################################

# Read the image to be analyzed
read_file_path = "D:\pycharmBao\pythonProject2\example\images\\x.png"

# Open the image file
image = Image.open(read_file_path)

# Resize the image to the specified size
image = image.resize((224, 224))

# Convert the image to a NumPy array
image_array = np.array(image)

# Based on the RGB display mode, extract the data of each channel
image_red_channel = image_array[:,:,0]
image_green_channel = image_array[:,:,1]
image_blue_channel = image_array[:,:,2]

# Generate an analysis image of size 224x224 based on the input image
out = '../images/input_image_aircraft_2_analyze'
Generate_preview_image(image_red_channel, image_green_channel, image_blue_channel, out)

#########################################################################################################



#########################################################################################################

# Normalize the R, G, and B channels of the analysis image individually
normal_image_red_channel = image_red_channel / 255
normal_image_green_channel = image_green_channel / 255
normal_image_blue_channel = image_blue_channel / 255

#########################################################################################################


#########################################################################################################

# Read the sensitivity matrices for the R, G, and B channels of the AlexNet network model
file_path_1 = './sensitivity_value_matrix_R.txt'
file_path_2 = './sensitivity_value_matrix_G.txt'
file_path_3 = './sensitivity_value_matrix_B.txt'

# Use the loadtxt function from NumPy to load the contents of a file into a NumPy array
sensitivity_value_R = np.loadtxt(file_path_1)
sensitivity_value_G = np.loadtxt(file_path_2)
sensitivity_value_B = np.loadtxt(file_path_3)


#########################################################################################################



#########################################################################################################

# Compute the sensitivity matrices of the R, G, and B channels for the input image
sensitivity_level_R = (ReLu_matrix(sensitivity_value_R)) * normal_image_red_channel
test_data_R = pd.DataFrame(sensitivity_level_R)
sensitivity_level_R = Max_Min_normalization(test_data_R)

sensitivity_level_G = (ReLu_matrix(sensitivity_value_G)) * normal_image_green_channel
test_data_R = pd.DataFrame(sensitivity_level_G)
sensitivity_level_G = Max_Min_normalization(test_data_R)

sensitivity_level_B = (ReLu_matrix(sensitivity_value_B)) * normal_image_blue_channel
test_data_B = pd.DataFrame(sensitivity_level_B)
sensitivity_level_B = Max_Min_normalization(test_data_B)
#########################################################################################################


#########################################################################################################

# Select the sensitive features of the R, G, and B channels
shou_max_matrix = np.zeros((224,224))
for i in range(224):
    for j in range(224):
        shou_max_matrix[i][j] = max(sensitivity_level_R[i][j], sensitivity_level_G[i][j], sensitivity_level_B[i][j])


# Plot a heatmap
heatmap = cv2.normalize(shou_max_matrix, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap_color_flipped = cv2.flip(heatmap_color, 1)
heatmap_rotated = cv2.rotate(heatmap_color_flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)
plt.imshow(cv2.cvtColor(heatmap_rotated, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save heatmap
save_path = r'../images/input_image_aircraft_2_SeiCoNet_heatmap.png'
cv2.imwrite(save_path, heatmap_rotated)






