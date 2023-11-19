import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms, datasets
from model import AlexNet


# Set the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perform initialization operations on the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Import the testing dataset
data_root = os.path.abspath(os.path.join(os.getcwd()))
image_path = os.path.join(data_root, "data_set")
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=transform)

# test_dataset = ImageFolder(root="", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load a pre-trained AlexNet model
model = AlexNet(num_classes=4).to(device)
model.load_state_dict(torch.load("./AlexNet.pth"))  # loading weights file
model.to(device)

# Set the model to evaluation mode
model.eval()

# Number of correctly predicted samples
correct = 0

# Total number of samples in the validation set
total = 0

# Disable gradient calculation during validation to save memory and computation time
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute accuracy
accuracy = 100 * correct / total
print("Accuracy: {:.2f}%".format(accuracy))