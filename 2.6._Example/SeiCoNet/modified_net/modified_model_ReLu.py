# Remove the ReLU layer and dropout layer in the original AlexNet network
import torch

#load weights file
path = './AlexNet.pth'
pretrained_dict = torch.load(path)

#modify the layer hierarchy correspondence
pretrained_dict["features.2.weight"] = pretrained_dict.pop("features.3.weight")
pretrained_dict["features.2.bias"] = pretrained_dict.pop("features.3.bias")

pretrained_dict["features.4.weight"] = pretrained_dict.pop("features.6.weight")
pretrained_dict["features.4.bias"] = pretrained_dict.pop("features.6.bias")

pretrained_dict["features.5.weight"] = pretrained_dict.pop("features.8.weight")
pretrained_dict["features.5.bias"] = pretrained_dict.pop("features.8.bias")

pretrained_dict["features.6.weight"] = pretrained_dict.pop("features.10.weight")
pretrained_dict["features.6.bias"] = pretrained_dict.pop("features.10.bias")

pretrained_dict["classifier.0.weight"] = pretrained_dict.pop("classifier.1.weight")
pretrained_dict["classifier.0.bias"] = pretrained_dict.pop("classifier.1.bias")

pretrained_dict["classifier.1.weight"] = pretrained_dict.pop("classifier.4.weight")
pretrained_dict["classifier.1.bias"] = pretrained_dict.pop("classifier.4.bias")

pretrained_dict["classifier.2.weight"] = pretrained_dict.pop("classifier.6.weight")
pretrained_dict["classifier.2.bias"] = pretrained_dict.pop("classifier.6.bias")

#save new weights file
torch.save(pretrained_dict, './modified_model_deReLu.pth')

