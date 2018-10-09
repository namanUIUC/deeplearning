import sys
import time
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms



def loader(batch_size):
    '''Training and Training dataset from CIFAR'''
    
    # Data Augmentation
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # Download Data
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    # Load as tensor
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    
    # Data Augmentation
    test_transform = transforms.Compose([transforms.ToTensor()])
    # Download Data
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    # Load as tensor
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
