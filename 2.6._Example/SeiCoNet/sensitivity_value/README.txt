This folder is primarily intended for symbolically executing AlexNet to obtain a symbolic model, including sensitivity matrices for the R, G, and B channels.

The "sensitivity_value_matrix_R.py" file is used to compute the sensitivity matrix for the R channel, the "sensitivity_value_matrix_G.py" file for the G channel, and the 
"sensitivity_value_matrix_B.py" file for the B channel. As the execution processes for these three files are similar, the following will focus on explaining the execution 
process of the "sensitivity_value_matrix_R.py" file.

In the first step, it is necessary to perform probability prediction on the image "Image R0 G0 B0.png" in the "Symbolic_Images" folder, and record the obtained classification
 probability values as an overall bias for the subsequent execution of the "sensitivity_value_matrix_B.py" file. During this process, it is important to note that in the "predict.py" file, 
when conducting probability prediction, the modified model "new_model.py" should be imported as AlexNet. This model has removed ReLU and modified the feature extraction 
function to average pooling. Additionally, at the "weights_path" location, add the path to the modified weight file "modified_model_deReLu_modweight.pth."

In the second step, after obtaining the overall bias for "new_model.py," symbolic execution is performed in the "sensitivity_value_matrix_R.py" file. Set the "folder_path" to the 
absolute path of the "picture_R" folder in the "Symbolic_Images" directory. Then, set the "json_path" to the absolute path of the "class_indices.json" file for classification indices.
 Next, set the "weights_path" to the absolute path of the "modified_model_deReLu_modweight.pth" weight file. Finally, set the first element value in the "cha" list to the overall 
bias obtained in the first step. By executing this file, the sensitivity matrix for the R channel can be obtained and saved in the "sensitivity_value_matrix_R.txt" file. The computation 
process for the G and B channels is similar, requiring only the adjustment of the "folder_path" to the respective "picture" path in the "Symbolic_Images" folder.