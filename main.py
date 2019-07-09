from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import os
from itertools import count
import time
import random
import numpy as np

from models.models import *
from models.preact_resnet import *

from torchvision.utils import save_image

if not torch.cuda.is_available():
    print('cuda is required but cuda is not available')
    exit()

#== parser start
parser = argparse.ArgumentParser(description='PyTorch')
# base setting 1: fixed
parser.add_argument('--job-id', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
# base setting 2: fixed
parser.add_argument('--test-batch-size', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--data-path', type=str, default='./dataset/')                    
# experiment setting
parser.add_argument('--dataset', type=str, default='mnist') 
parser.add_argument('--data-aug', type=int, default=0) 
parser.add_argument('--model', type=str, default='LeNet') 
# method setting
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--ssize', type=int, default=64)
parser.add_argument('--method', type=int, default=0) 
                    # --method=0: standard
                    # --method=1: q-SGD 
args = parser.parse_args()                    
#== parser end
data_path = args.data_path + args.dataset
if not os.path.isdir(data_path):
    os.makedirs(data_path)

result_path = './results/'    
if not os.path.isdir(result_path):
    os.makedirs(result_path)
result_path += args.dataset + '_' + str(args.data_aug) + '_' + args.model
result_path += '_' + str(args.method) + '_' + str(args.batch_size)
if args.method != 0:
    result_path += '_' + str(args.ssize) 
result_path += '_' + str(args.job_id)
filep = open(result_path + '.txt', 'w')
with open(__file__) as f: 
    filep.write('\n'.join(f.read().split('\n')[1:]))
filep.write('\n\n')    

out_str = str(args)
print(out_str)
filep.write(out_str + '\n') 

if args.seed is None:
  args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.enabled = True

out_str = 'initial seed = ' + str(args.seed)
print(out_str)
filep.write(out_str + '\n\n')

#===============================================================
#=== dataset setting
#===============================================================
kwargs = {}
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_Sampler = None
test_Sampler = None
Shuffle = True
if args.dataset == 'mnist':
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 50
    if args.data_aug == 1:        
        end_epoch = 200 
        train_transform = transforms.Compose([
                            transforms.RandomCrop(28, padding=2),
                            transforms.RandomAffine(15, scale=(0.85, 1.15)),
                            transforms.ToTensor()       
                       ])                
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'cifar10':
    nh = 32
    nw = 32
    nc = 3
    num_class = 10 
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200 
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    train_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'cifar100':
    nh = 32
    nw = 32
    nc = 3
    num_class = 100
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200    
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    train_data = datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root=data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'svhn':
    nh = 32
    nw = 32
    nc = 3
    num_class = 10
    end_epoch = 50    
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    train_data = datasets.SVHN(data_path, split='train', download=True, transform=train_transform)
    test_data = datasets.SVHN(data_path, split='test', download=True, transform=test_transform)    
elif args.dataset == 'fashionmnist':
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 20
    if args.data_aug == 1:
        end_epoch = 200       
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]) 
    train_data = datasets.FashionMNIST(data_path, train=True, download=True, transform=train_transform)
    test_data = datasets.FashionMNIST(data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'kmnist':
    nh = 28
    nw = 28
    nc = 1
    num_class = 10
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose([
                            transforms.RandomCrop(28, padding=2),
                            transforms.ToTensor()       
                       ])
    train_data = datasets.KMNIST(data_path, train=True, download=True, transform=train_transform)
    test_data = datasets.KMNIST(data_path, train=False, download=True, transform=test_transform)
elif args.dataset == 'semeion':
    nh = 16
    nw = 16
    nc = 1
    num_class = 10 # the digits from 0 to 9 (written by 80 people twice)    
    end_epoch = 50
    if args.data_aug == 1:
        end_epoch = 200
        train_transform = transforms.Compose([
            transforms.RandomCrop(16, padding=1),
            transforms.RandomAffine(4, scale=(1.05, 1.05)),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])    
    train_data = datasets.SEMEION(data_path, transform=train_transform, download=True) 
    test_data = train_data    
    random_index = np.load(data_path+'/random_index.npy')
    train_size = 1000    
    train_Sampler = SubsetRandomSampler(random_index[range(train_size)])
    test_Sampler = SubsetRandomSampler(random_index[range(train_size,len(test_data))])
    Shuffle = False
elif args.dataset == 'fakedata':
    nh = 24
    nw = 24
    nc = 3
    num_class = 10  
    end_epoch = 50   
    train_size = 1000
    test_size = 1000
    train_data = datasets.FakeData(size=train_size+test_size, image_size=(nc, nh, nw), num_classes=num_class, transform=train_transform)
    test_data  = train_data 
    train_Sampler = SubsetRandomSampler(range(train_size))
    test_Sampler = SubsetRandomSampler(range(train_size,len(test_data)))
    Shuffle = False    
else: 
    print('specify dataset')
    exit()   
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,      sampler=train_Sampler, shuffle=Shuffle, **kwargs)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=args.test_batch_size, sampler=test_Sampler,  shuffle=False,   **kwargs)

#===============================================================
#=== model setting
#===============================================================
if args.model == 'LeNet':
    model = LeNet(nc, nh, nw, num_class).cuda()
elif args.model == 'PreActResNet18':
    model = PreActResNet18(nc, num_class).cuda()
elif args.model == 'Linear' or args.model == 'SVM':
    dx = nh * nw * nc     
    model = Linear(dx, num_class).cuda()
else:
    print('specify model')
    exit() 
    
#===============================================================
#=== utils def
#===============================================================
def lr_decay_func(optimizer, lr_decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1   
    return optimizer    
def lr_scheduler(optimizer, epoch, lr_decay=0.1, interval=10):
    if args.data_aug == 0:
        if epoch == 10 or epoch == 50:
            optimizer = lr_decay_func(optimizer, lr_decay=lr_decay) 
    if args.data_aug == 1:
        if epoch == 10 or epoch == 100:
            optimizer = lr_decay_func(optimizer, lr_decay=lr_decay)                   
    return optimizer

class multiClassHingeLoss(nn.Module):
    def __init__(self):
        super(multiClassHingeLoss, self).__init__()
    def forward(self, output, y):
        index = torch.arange(0, y.size()[0]).long().cuda()
        output_y = output[index, y.data.cuda()].view(-1,1)
        loss = output - output_y + 1.0 
        loss[index, y.data.cuda()] = 0
        loss[loss < 0]=0
        loss = torch.sum(loss, dim=1) / output.size()[1]
        return loss 
hinge_loss = multiClassHingeLoss()
    
#===============================================================
#=== train optimization def
#===============================================================
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  
ssize = args.ssize
def train(epoch):
    global optimizer, ssize
    model.train()
    optimizer = lr_scheduler(optimizer, epoch)  
    
    train_acc_prev = pl_result[epoch-1, 0, 1]
    if train_acc_prev >= 99.5 and ssize > 4:
        ssize = 4        
        optimizer = lr_decay_func(optimizer, lr_decay=0.5)
    elif train_acc_prev >= 95 and ssize > 8:
        ssize = 8
    elif train_acc_prev >= 90 and ssize > 16:
        ssize = 16    
    elif train_acc_prev >= 80 and ssize > 32:
        ssize = 32 
        
    for batch_idx, (x, y) in enumerate(train_loader):
        bs = y.size(0) 
        x = Variable(x.cuda())
        y = Variable(y.cuda())     
        h1 = model(x)
        if args.model == 'SVM':
            cr_loss = hinge_loss(h1, y)
        else:        
            cr_loss = F.cross_entropy(h1, y, reduction='none')
        if args.method == 0 or ssize >= bs:
            loss = torch.mean(cr_loss)             
        elif args.method == 1:                
            loss = torch.mean(torch.topk(cr_loss, min(ssize, bs), sorted=False, dim=0)[0]) 
        else:
            print('specify method')
            exit()                              
        optimizer.zero_grad() 
        loss.backward()        
        optimizer.step()  
     
    optimizer.zero_grad() 


#===============================================================
#=== train/test output def
#===============================================================    
def output(data_loader):
    if data_loader == train_loader:    
        model.train()
    elif data_loader == test_loader:
        model.eval()
    total_loss = 0    
    total_correct = 0      
    total_size = 0   
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = Variable(x.cuda()), Variable(y.cuda())
        h1 = model(x)
        y_hat = h1.data.max(1)[1]
        if args.model == 'SVM':
            total_loss += torch.mean(hinge_loss(h1, y)).item() * y.size(0)
        else:                
            total_loss += F.cross_entropy(h1, y).item() * y.size(0)
        total_correct += y_hat.eq(y.data).cpu().sum()                
        total_size += y.size(0)    
    # print
    total_loss /= total_size 
    total_acc = 100. * float(total_correct) / float(total_size)  
    if data_loader == train_loader:    
        out_str = 'tr_l={:.3f} tr_a={:.2f}:'.format(total_loss, total_acc) 
    elif data_loader == test_loader:
        out_str = 'te_l={:.3f} te_a={:.2f}:'.format(total_loss, total_acc)            
    print(out_str, end=' ')
    filep.write(out_str + ' ') 
    return (total_loss, total_acc)  

#===============================================================
#=== start computation
#===============================================================    
#== for plot
pl_result = np.zeros((end_epoch+1, 3, 2))  # epoch * (train, test, time) * (loss , acc) 
#== main loop start
time_start = time.time()
for epoch in count(0):
    out_str = str(epoch)
    print(out_str, end=' ') 
    filep.write(out_str + ' ')
    if epoch >= 1:
        train(epoch)
    pl_result[epoch, 0, :] = output(train_loader)
    pl_result[epoch, 1, :] = output(test_loader)
    time_current = time.time() - time_start
    pl_result[epoch, 2, 0] = time_current
    np.save(result_path + '_' + 'pl', pl_result)    
    out_str = 'time={:.1f}:'.format(time_current) 
    print(out_str)    
    filep.write(out_str + '\n')   
    if epoch == end_epoch:
        break