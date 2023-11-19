The first step involves modifying the parameters of the well-trained AlexNet in the file "modified_net," implementing the SimReLU effect mentioned in the paper.

In the second step, within the "sensitivity_value" folder, symbolic execution is performed on the AlexNet network. After execution, sensitivity matrices for the R, G, and B 
channels of the image can be obtained and stored in documents named "sensitivity_value_matrix_R.txt," "sensitivity_value_matrix_G.txt," and "sensitivity_value_matrix_B.txt."

The third step involves generating a saliency map in the "saliency_map_result.py" file. When executing this file, import the absolute path of the target image at the "read_file_path"
 location. Import the sensitivity matrices for the R, G, and B channels at the "file_path_1," "file_path_2," and "file_path_3" locations, respectively. Finally, set the "save_path" to specify
 the path for saving the generated saliency map.

Of course, you can start directly from the third step since this folder already contains all the necessary files for the execution of the third step. Simply follow the instructions in the third 
step to set the paths for each file.