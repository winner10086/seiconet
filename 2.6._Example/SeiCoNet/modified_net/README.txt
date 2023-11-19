This folder is used to modify the AlexNet weight file to implement SimReLU, which is a prerequisite for symbolically executing this network.

In the first step, within the "modified_model_ReLu.py" file, all ReLU activations in the original AlexNet network are removed. You need to import the absolute path 
of the well-trained AlexNet weight file "AlexNet.pth" at the "path" location in the file. After removing ReLU, the new weight file is named "modified_model_deReLu.pth."

In the second step, all ReLU layers are removed from the original AlexNet structure, creating a new AlexNet structure as shown in the file "new_model.py."

In the third step, within the "modified_model_weights.py" file, SimReLu is performed. Set the path in the "model.load_state_dict" to the path of the "modified_model_deReLu.pth" 
generated in the first step.
