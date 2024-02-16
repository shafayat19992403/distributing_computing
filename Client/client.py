import warnings
from collections import OrderedDict


import argparse  # Import argparse
from utils.util_class import CNNModel
from utils.util_function import *


import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# #############################################################################
# 0. Parse command line arguments
# #############################################################################

parser = argparse.ArgumentParser(description="Federated Learning with Flower and PyTorch")
parser.add_argument("--server_address", type=str, default="127.0.0.1:3000", help="Address of the FL server")
parser.add_argument("--threshold_loss", type=float, default=500, help="Threshold for loss change")
parser.add_argument("--threshold_accuracy", type=float, default=0.02, help="Threshold for accuracy change")
parser.add_argument("--data_path", type=str, default="./data", help="Path to CIFAR-10 data")
parser.add_argument("--poison_rate", type=float, default=0, help="Rate of poisoning in the dataset (0 to 1)")
parser.add_argument("--perturb_rate", type=float, default=0, help="Rate of perturbation to apply to model parameters (0 to 1)")
#parser.add_argument("--isMal", type=bool, default=False, help="Is the client malicious")

args = parser.parse_args()

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# def load_data():
#     """Load CIFAR-10 (training and test set)."""
#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10("./data", train=True, download=True, transform=trf)
#     testset = CIFAR10("./data", train=False, download=True, transform=trf)
#     return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
# def load_data(data_path):
#     """Load CIFAR-10 (training and test set)."""
#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10(data_path, train=True, download=True, transform=trf)
#     testset = CIFAR10(data_path, train=False, download=True, transform=trf)
#     return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
# def poison_dataset(dataset, poison_rate):
#     """Poison the dataset by flipping the label of a certain fraction of samples."""
#     n = len(dataset)
#     idxs = list(range(n))
#     np.random.shuffle(idxs)
#     idxs = idxs[:int(poison_rate * n)]
#     for i in idxs:
#         x, y = dataset[i]
#         y = (y + 1) % 10
#         dataset.targets[i] = y
#     return dataset

def load_data(data_path, poison_rate):
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(data_path, train=True, download=True, transform=trf)
    testset = CIFAR10(data_path, train=False, download=True, transform=trf)
    
    if poison_rate > 0:
        # Assuming you have a function to poison the dataset
        trainset = poison_CIFAR10_dataset(trainset, poison_rate=poison_rate)
    
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)



# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
# net = Net().to(DEVICE)
# trainloader, testloader = load_data()

net = Net().to(DEVICE)
trainloader, testloader = load_data(args.data_path, args.poison_rate)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, threshold_loss=500, threshold_accuracy=0.02,perturb_rate=0.0):
        super().__init__()
        self.previous_loss = None
        self.previous_accuracy = None
        self.threshold_loss = threshold_loss
        self.threshold_accuracy = threshold_accuracy
        self.perturb_rate = perturb_rate

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def perturb_parameters(self, parameters, perturb_rate):
        # Implement parameter perturbation logic here
        perturbed_parameters = [param + np.random.normal(scale=perturb_rate, size=param.shape) for param in parameters]
        return perturbed_parameters

    def fit(self, parameters, config):
        if self.perturb_rate > 0:
            parameters = self.perturb_parameters(parameters, perturb_rate=self.perturb_rate)
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    
    def validity(self, parameters, config):
        print("Inside checking validity")
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        validity =  1
        if self.previous_loss is not None and self.previous_accuracy is not None:
            if loss > self.previous_loss + self.threshold_loss or accuracy < self.previous_accuracy - self.threshold_accuracy:
                validity = 0
        return validity

    def evaluate(self, parameters, config):  # validate evaluate
        print(config)
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        validity =  1
        try:
            if config["mode"] == 'validate':
                print("checking Validation")
                if self.previous_loss is not None and self.previous_accuracy is not None and args.poison_rate == 0:
                    if loss > self.previous_loss + self.threshold_loss or accuracy < self.previous_accuracy - self.threshold_accuracy:
                        validity = 0
                
                print("validity"+str(validity))
                return loss, len(testloader.dataset), {"validity": validity}

            else:
                print('checking evaluation')    
                self.previous_loss = loss
                self.previous_accuracy = accuracy
                print("loss: "+ str(loss) + " accuracy: "+ str(accuracy)) 
                return loss, len(testloader.dataset), {"accuracy": accuracy}
        except:
            self.previous_loss = loss
            self.previous_accuracy = accuracy
            print("loss: "+ str(loss) + " accuracy: "+ str(accuracy)) 
            return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
print("ok")
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:3000",
#     client=FlowerClient(),
# )
print(args.server_address,args.threshold_loss,args.threshold_accuracy)
fl.client.start_numpy_client(
    server_address=args.server_address,
    client=FlowerClient(threshold_loss=args.threshold_loss, threshold_accuracy=args.threshold_accuracy,perturb_rate=args.perturb_rate),
)
print(args.server_address,args.threshold_loss,args.threshold_accuracy)