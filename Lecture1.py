'''

Created by 'Major' Myles Z. MA, 21-11-25, 19:50, Xiqing, TIANJIN
- UTF-8 -


### _______________________________ LEARN THE BASICS __________________________________ ###

Most ML workflows include: 
working with data; creating models; optimizing model parameters; saving the trained models.

This lecture introduces you to a complete workflow inplemented by PyTorch.

We’ll use the FashionMNIST dataset to train a neural network that predicts if an input 
image belongs to one of the following classes: T-shirt/top, Trouser, Pullover, Dress, Coat, 
Sandal, Shirt, Sneaker, Bag, or Ankle boot.


'''


### --------------------------------- 0. QUICKSTART ----------------------------------- ###
#   This section runs through the API for common tasks in machine learning. 
#   Refer to the links in each section to dive deeper.
#
#   Working with data
#   PyTorch has two primitives to work with data: torch.utils.data.DataLoader and 
# torch.utils.data.Dataset. 
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable 
# around the Dataset.



#from os import pread
import torch
from torch import device, mode, nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt



train_data = datasets.FashionMNIST(
    #root='/Users/majorma/Documents/vscode/Pytorch_tutorial/data'
    root='E:\Z_Source_Codes\Vscode_Codes\Python\PytorchTutorial\LearnPytorch\data',
    train=True,
    download=True,
    transform=ToTensor()
)


test_data = datasets.FashionMNIST(
    #root='/Users/majorma/Documents/vscode/Pytorch_tutorial/data'
    root='E:\Z_Source_Codes\Vscode_Codes\Python\PytorchTutorial\LearnPytorch\data',
    train=False,
    download=True,
    transform=ToTensor()
)


#   We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset, 
# and supports automatic batching, sampling, shuffling and multiprocess data loading. 
#   
#   Here we define a batch size of 64, i.e. each element in the dataloader iterable will return 
# a batch of 64 features and labels.
batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)



for X, y in test_dataloader:
    print('Shape of X [N, C, H, W]: ', X.shape)
    print('Shape of y [N, C, H, W]: ', y.shape)
    break


###     Creating Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training.")

class MajorNet(nn.Module):
    def __init__(self):
        super(MajorNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_task  = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_task(x)
        return logits

#   这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
model = MajorNet().to(device)
print(model)




''' TODO: Change Net style into a common style. '''
### ______________________________________________________________________ ###
class MajorNetAnother(nn.Module):
    def __init__(self, dim_input, numb_classes, mid_out):
        super(MajorNetAnother, self).__init__()
        self.faltten = nn.Flatten()
        self.linear1 = nn.Linear(dim_input * dim_input, mid_out)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(mid_out, numb_classes)
        self.linear2 = nn.Linear(mid_out, mid_out)
    
    def forward(self, x):
        out = self.faltten(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)

### ______________________________________________________________________ ###



###     Optimizing Model Para
#   Need a loss func and an optimizer!
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.3)



''' TODO 尚未运行'''
#   In a single training loop, the model makes predictions on the training dataset 
# (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

    # Pred. error
    pred = model(X)
    loss = loss_func(pred, y)

    # BP
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")



### We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {100*correct:>0.1f}%, avg loss: {test_loss:>8f} \n")



### The training process is conducted over several iterations (epochs). During each epoch, 
# the model learns parameters to make better predictions. We print the model’s accuracy and loss 
# at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------------------------")
    train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
print("DONE!")



###     Saving Models
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


###     Loading Models
model = MajorNet()
model.load_state_dict(torch.load('model.pth'))


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')