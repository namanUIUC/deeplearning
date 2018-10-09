import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Dropout2d(p=0.5),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.AvgPool2d(kernel_size=1, stride=1),
                                      )

        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
