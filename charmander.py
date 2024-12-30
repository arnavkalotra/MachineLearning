import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download training data from open datasets.         boolean statements to sort and form a tensor
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.                    same thing for the test data 
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64 #                                           this batch 64 means it will return 64 different features of the data when called, samples in chunks to reduce load

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")                               #essentially returns the size of the resulting tensors
    break
#                                                           N- number of batches, C- number of channels (1 for grayscale), H- height, W- width

#                                                       Definining  A NUeral Network




# Get cpu, gpu or mps device for training.                  similar to using qiskit servers
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):                         #base class for all nueral networks
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                     #converts 2d image into 1x1 arrays


                                #the nueron connections
#                                                                   Each input value gets multiplied by a weight and added by a bias term, a sum of those nuerons results in Z (value ager linear transf)
                                                          #cost function essentially 
        self.linear_relu_stack = nn.Sequential(                 #this ReLu introduces non lineaerity , gives a rigor to the nuerons
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )                                                   #wieghts get updated using learning rate and gradients of the loss, ( machine learning cost funnction)
    def forward(self, x):        #how input data passes thru the network
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)                                          #network outputs logits, usually converted into probabilities using SOFTMAX FUNCTION
        return logits

model = NeuralNetwork().to(device)
print(model)