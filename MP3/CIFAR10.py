import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import torch.nn as nn
import torch

import argparse

from models.LeNet import *
from models.Network import *
from utils import loader
from torch.autograd import Variable


# CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser = argparse.ArgumentParser(description="CIFAR10")
parser.add_argument('--lr', default=0.0011, type=float, help='learning rate')
parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
parser.add_argument('--BatchSize', default=100, type=int, help='batch size')
parser.add_argument('--cuda', default=False, type=bool, help='whether cuda is in use')
parser.add_argument('--load', default=False, type=bool, help='whether to load the model')
parser.add_argument('--mc_sample', default=50, type=int, help='sample size of the prediction')
parser.add_argument('--mode', default='blank', type=str, help='mode : {eval, mc}')
args = parser.parse_args()


class Manager(object):
    def __init__(self):
        self.model = None
        self.lossfunc = None
        self.optimizer = None
        self.scheduler = None
        self._build_model()

    def _build_model(self):
        
        self.model = Net()
        
        if args.load:
            self.load()
        if args.cuda:
            self.model = self.model.cuda()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.lossfunc = nn.CrossEntropyLoss()

    def train(self, train_set):
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(train_set):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.lossfunc(output, target)
            loss.backward()

            self.optimizer.step()
            train_loss += float(loss.data.cpu().numpy())
            prediction = torch.max(output, 1)

            total += target.size(0)
            train_correct += np.sum((prediction[1].data).cpu().numpy() == (target.data).cpu().numpy())

        return train_loss, train_correct / total

    def test(self, test_set):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        # with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_set):
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = self.model(data)
            loss = self.lossfunc(output, target)
            test_loss += float(loss.data.cpu().numpy())
            prediction = torch.max(output, 1)
            total += target.size(0)
            test_correct += np.sum((prediction[1].data).cpu().numpy() == (target.data).cpu().numpy())

        return test_loss, test_correct / total

    def save(self):
        torch.save(self.model.state_dict(), './ckpt/params.ckpt')
    
    def load(self):
        self.model.load_state_dict(torch.load('./ckpt/params.ckpt'))

    def evaluate(self, train_set, test_set):

        for epoch in range(1, args.epoch + 1):
            self.scheduler.step(epoch)
            # Adam
            if(epoch > 12):
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        state = self.optimizer.state[p]
                        if(state['step'] >= 1024):
                            state['step'] = 1000

            train_result = self.train(train_set)
            test_result = self.test(test_set)

            msg = "Epoch: {0:>2}, Training Loss: {1:>6.4f}, Training Acc: {2:>6.3%}, Validation Loss: {3:>6.4f}, Validation Acc: {4:>6.3%}"
            print(msg.format(epoch, train_result[0], train_result[1], test_result[0], test_result[1]))

            if epoch == args.epoch:
                self.save()
                
    def mc(self, test_set):
        
        acc = []
        for sample in range(5, args.mc_sample+5, 5):
            print(sample)
            correct = 0
            total = 0

            for batch_num, (data, target) in enumerate(test_set):
                data, target = Variable(data).cuda(), Variable(target).cuda()

                flag = 1
                for m in range(sample):
                    self.model.train()
                    output = self.model(data)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    probs = probs.unsqueeze(2)
                    if flag:
                        flag = 0
                        new_probs = probs
                    else:
                        new_probs = torch.cat((new_probs, probs), dim=2)
                
                
                prediction = torch.max(torch.mean(new_probs,dim=2), 1)
                total += target.size(0)
                correct += np.sum((prediction[1].data).cpu().numpy() == (target.data).cpu().numpy())
            msg = "Samples: {0:>2},  Accuracy: {1:>6.3%}"
            print(msg.format(int(sample/5), correct/total))
            
            acc.append(correct/total) 
            
        return acc    
        


def main():

    train_set, test_set = loader(args.BatchSize)

    mgr = Manager()
    
    if args.mode == 'eval':
        mgr.evaluate(train_set, test_set)
    elif args.mode == 'mc':
        acc = mgr.mc(test_set)
        import pickle
        with open("acc.txt", "wb") as fp:   #Pickling
            pickle.dump(acc, fp)
    else:
        print('choose one of the mode : eval, mc')


if __name__ == '__main__':
    main()
