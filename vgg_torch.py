'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
#import cv2
from PIL import Image

######################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

######################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import os
import datetime
####################################################

patb = 'dataset_small_'
train_dir = './dataset_small/train/'

#eval_dir  = './dataset_small/test/'
eval_dir  = './dataset_small/test2/'

result_dir  = './result/'
ti_now = datetime.datetime.now()
ti_clock =  ti_now.strftime("%m_%d_%H_%M")

result_csv = result_dir+patb+ti_clock+'.csv'
####################################################

'''
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

'''


#################################################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return

class MyDataset(data.Dataset):
    def __init__(self, root, transform=None):
        img_p = [os.path.join(root, x) for x in os.listdir(root) if is_image_file(x)]
        self.root = root
        self.transform = transform
        self.img_p = img_p

    def __getitem__(self, index):
        target = self.img_p[index]
        img = Image.open(target).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.img_p)


#224x224
print('\n [INFO] Load data.. \n')

'''
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
'''

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

####################################################
# torchvision.datasets.ImageFolder(root, transform=None, target_transform=None
                               # , loader=<function default_loader>, is_valid_file=None)
trainset    = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)

testset      = MyDataset(root=eval_dir, transform=transform_eval)
testloader   = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


classes = ('bedroom', 'coast', 'forest', 'highway', 'insidecity'
         , 'kitchen', 'livingroom', 'mountain', 'office', 'opencountry'
         , 'street', 'suburb', 'tallbuilding')


def imshow(img):
    img = img/2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def check_set():
    print("\n\n\n--------- Check trainset  \n ")
    #print(trainset.class_to_idx)
    #print(trainset.imgs[0][0])

    for ind, (inputs, labels) in enumerate(trainloader, 0):
        if(ind < 50):
            print ("images size", inputs.size(), "labels: ", labels)
            #print ("inputs: ", inputs, "labels: ", labels)
            #imshow(torchvision.utils.make_grid(inputs))

    print("\n\n\n--------- Check testset  \n ")

    for ind, (inputs, target) in enumerate(testloader, 0):
        if(ind < 20):
            print ("images size", inputs.size(), "target: ", target)
            #print ("inputs: ", inputs, "target: ", target)
            #imshow(torchvision.utils.make_grid(inputs))




########################
net = models.vgg19()
#net = models.vgg19_bn(num_classes=13)
#net = models.vgg19_bn(pretrained=True)

if (torch.cuda.is_available()):
    net.cuda()
    torch.backends.cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.0001)

def train_img(load_mdl):
    print('\n [INFO] START Training \n')
    if(load_mdl):
        net.load_state_dict(torch.load('./mdl_state/mdl_save'))

    net.train()
    iters = 50
    for epoch in range(iters):
        running_loss = 0.0
        print("\n----------New Epoch ", epoch)
        for ind, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #_, predicted = torch.max(outputs.data, 1)
            # compute gradient and do SGD step

            '''
            if(ind == 5):
                print(" predicted=", predicted[0], "labels=", labels[0])

            if( predicted[0] !=  labels[0]):
                running_loss += 1
            if ind % 100 == 99:    # print every 2000 mini-batches
                running_loss = running_loss/100
                print(" running_loss=", running_loss, " epoch=", epoch, "ind=", ind)
                running_loss = 0.0
            '''

            # print statistics
            if(ind % 100 == 99):
                print(" loss.data=", loss.data, "labels=", labels[0])


    torch.save(net.state_dict(), './mdl_state/mdl_save')
    print('\n [INFO] Finished Training \n')

###############################################


def eval_img(load_mdl):
    print('\n [INFO] Start Evaluation..... \n')
    if(load_mdl):
        net.load_state_dict(torch.load('./mdl_state/mdl_save'))
    net.eval()
    fo = open(result_csv, "w")
    fo.write("id, label \n")

    for ind, (inputs, target) in enumerate(testloader, 0):
        inputs = inputs.cuda()

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        maxp = predicted.tolist()
        filen = ''.join(target)
        filen = filen.replace(eval_dir, '')
        filen = filen.replace('.jpg', '')
        #line_str = str(filen)+", "+str(classes[maxp[0]])+"\n"
        line_str = str(filen)+", "+str(maxp[0])+"\n"
        fo.write(line_str)
        #if(ind < 10):
            #print("\n ",ind, "-th outputs ", outputs)
            #print("\n ",ind, "-th pre= ", predicted)
            #print("\n ",line_str)
    fo.close()
    print('\n [INFO] Finished Evaluation..... \n')

#######################
#check_set()

load_mdl = 1

train_img(load_mdl)

#eval_img(load_mdl)

