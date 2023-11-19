from PIL import Image
import os

# Function to extract the numerical part of the file name
def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

# Folder paths
original_folder_path = r"D:\pycharmBao\pythonProject2\Comparison experiments\Experimental Images\frog\dual-target images\Annotated images\SeiCoNet"
annotated_folder_path = r"D:\pycharmBao\pythonProject2\Comparison experiments\Experimental Images\frog\dual-target images\Annotated images\SeiCoNet"

# List all files in the folders and sort them
original_image_files = sorted(os.listdir(original_folder_path), key=sort_func)
annotated_image_files = sorted(os.listdir(annotated_folder_path), key=sort_func)

# Initialize a counter
count_greater_than_0_5 = 0

for i in range(len(original_image_files)):
    # Open the original image and annotated image files in order
    original_image = Image.open(os.path.join(original_folder_path, original_image_files[i]))
    annotated_image = Image.open(os.path.join(original_folder_path, annotated_image_files[i]))

    # Get the width and height of the images
    width, height = original_image.size

    # Count the number of object pixels in the original image
    count_object = 0

    # Count the total object pixels
    for x in range(width):
        for y in range(height):
            pixel = original_image.getpixel((x, y))
            if pixel[0] <= 250 and pixel[1] <= 250 and pixel[2] <= 255:
                count_object += 1

    # Count the annotated pixels in the object
    count_annotated = 0

    # Iterate through each pixel in the image
    for x in range(width):
        for y in range(height):
            pixel = annotated_image.getpixel((x, y))
            if pixel[0] <= 30 and pixel[1] <= 30 and pixel[2] >= 100:
                count_annotated += 1

    percentage = count_annotated / count_object
    image_name = original_image_files[i]
    print(f"Image: {image_name},   count_annotated:{count_annotated},  count_object:{count_object},  Percentage: {percentage}")

    # If the percentage is greater than 0.5, increment the counter
    if percentage > 0.5:
        count_greater_than_0_5 += 1

print("Number of images with percentage > 0.5:", count_greater_than_0_5)
