from PIL import Image
import os

# Define the path to the folder to be processed
folder_path = "D:\pycharmBao\pythonProject2\Max_3_dog\images\gou"

# Retrieve a list of all files in the folder
file_list = os.listdir(folder_path)

# Iterate through the list of files
for file_name in file_list:
    # Construct the full path of the file
    file_path = os.path.join(folder_path, file_name)

    # Open the image file
    image = Image.open(file_path)

    # Resize the image to 224x224
    resized_image = image.resize((224, 224))

    # Save the resized image back to the original file
    resized_image.save(file_path)

    # Close the image file
    image.close()

print("Image resizing completed!")