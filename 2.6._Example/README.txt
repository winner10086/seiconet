This folder contains three files, namely "AlexNet model", "experimental result", and "SeiCoNet". Let's delve into their respective details.

Firstly, let's introduce the main functionalities of each folder:

1. The folder "AlexNet model" includes the AlexNet neural network model used in the experiments described in section 2.6 of the paper. It comprises the trained weights of the network model after experimentation 
("AlexNet.pth"), along with the code files for training ("train.py") and testing ("test.py") the network model. Additionally, there are code files for image prediction using the network model ("predict.py").

2. The folder "experimental result" includes the images used in the experiments described in section 2.6 of the paper.

3. The folder "SeiCoNet" is used for symbolically executing the convolutional neural network trained in the "AlexNet model" folder and generating the final saliency map.

Secondly, let's explain how these three folders work together. The main execution process between each folder is described here:

When there is a trained AlexNet model, obtaining a saliency map for an image involves three steps. The first step is to input the image into the network model in the "AlexNet model" folder for category prediction, 
obtaining the image's category. The second step is to symbolically execute the network model in the "SeiCoNet" folder to obtain the sensitivity matrix for that category. Of course, the obtained sensitivity matrix is 
applicable to all images classified as that category. The third step is to calculate the final sensitivity matrix corresponding to the selected image by performing computations with the obtained sensitivity matrix in 
the "SeiCoNet" folder. This final sensitivity matrix is visualized to generate the saliency map.

More detailed instructions for each folder's operations can be found in their respective "README.txt" documents. However, you can also directly execute the third step since the folders already contain all the necessary 
data for this step; you just need to set the paths accordingly.