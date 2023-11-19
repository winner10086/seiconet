This folder contains all the files for a well-trained AlexNet neural network model, including the code files for training ("train.py") and testing ("test.py") this network.

The file "AlexNet.pth" comprises all the parameters of the trained network and serves as the weight file for this network.

The "class_indices.json" file contains the category indices learned by the model during the training process along with their corresponding class labels.

The file "model.py" encompasses the framework for the AlexNet network.

In the "predict.py" file, image category prediction can be performed. To do so, import the absolute path of the image to be determined at the "img_path" location in the file. 
Import the "class_indices.json" file at the "json_path" location and the "AlexNet.pth" file at the "weights_path" location. Afterward, you can obtain the category probabilities.