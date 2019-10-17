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
import numpy as np

import os
import datetime
####################################################

patb = 'dataset'
#train_dir = './dataset_small/train/'
train_dir = './dataset/train/'

#eval_dir  = './dataset_small/test/'
eval_dir = './dataset/test/'

result_dir  = './result/'
ti_now = datetime.datetime.now()
ti_clock =  ti_now.strftime("%m_%d_%H_%M")

result_csv = result_dir+patb+'_'+ti_clock+'.csv'

print(' train=', train_dir,'\n eval=', eval_dir, '\n result= ', result_csv)

#################################################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return

class MyDataset(data.Dataset):
    def __init__(self, root, transform=None):
        eval_lst = os.listdir(root)
        eval_lst.sort()
        img_p = [os.path.join(root, x) for x in eval_lst if is_image_file(x)]
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


#################################################

transform_train1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_train2 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
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
print('\n [INFO] Load data..')

trainset    = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train1)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=False)

testset      = MyDataset(root=eval_dir, transform=transform_eval)
testloader   = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


classes = ('bedroom', 'coast', 'forest', 'highway', 'insidecity'
         , 'kitchen', 'livingroom', 'mountain', 'office', 'opencountry'
         , 'street', 'suburb', 'tallbuilding')


def check_set():
    print("\n\n\n--------- Check trainset  \n ")

    for ind, (inputs, labels) in enumerate(trainloader, 0):
        if(ind < 25):
            print ("images size", inputs.size(), "labels: ", labels)
            #print ("inputs: ", inputs, "labels: ", labels)
    print("\n\n\n--------- Check testset  \n ")

    for ind, (inputs, target) in enumerate(testloader, 0):
        if(ind < 25):
            print ("images size", inputs.size(), "target: ", target)


########################
#net = models.vgg19()
net = models.vgg19(pretrained=True)
#net = models.vgg19_bn(pretrained=True)

if (torch.cuda.is_available()):
    net.cuda()
    torch.backends.cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


########################
def train_img(load_mdl):
    print('\n [INFO] START Training ')
    if(load_mdl):
        net.load_state_dict(torch.load('./mdl_state/mdl_save'))

    net.train()

    iters = 2
    loss_values = []
    for epoch in range(iters):
        print("----------New Epoch ", epoch)
        running_loss = 0.0
        for ind, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.cuda()
            labels = labels.cuda()
            # compute gradient and do SGD step
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #if(epoch % 5 == 4 and ind % 10 == 9):
            #    print(" loss.data=", loss.data, "labels=", labels[0])
            running_loss += loss.item()
        loss_values.append(running_loss)
        #print(" running_loss=", running_loss)
        #print(" loss_values=", loss_values)

    #torch.save(net.state_dict(), './mdl_state/mdl_save')
    print('\n [INFO] Finished Training ')
    fo = open('./result/train_1_loss.csv', "w")
    fo.write("epoch,loss\n")

    for idx in range(iters):
        line_str = str(idx)+","+str(loss_values[idx])
        fo.write(line_str)
    fo.close()


###############################################
def eval_img(load_mdl):
    print('\n [INFO] Start Evaluation.....')
    if(load_mdl):
        net.load_state_dict(torch.load('./mdl_state/mdl_save'))
    net.eval()
    fo = open(result_csv, "w")
    fo.write("id,label\n")

    for ind, (inputs, target) in enumerate(testloader, 0):
        inputs = inputs.cuda()

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        maxp = predicted.tolist()
        filen = ''.join(target)
        filen = filen.replace(eval_dir, '')
        filen = filen.replace('.jpg', '')
        line_str = str(filen)+","+str(classes[maxp[0]])+"\n"
        #line_str = str(filen)+", "+str(maxp[0])+"\n"
        fo.write(line_str)

    fo.close()
    print('\n [INFO] Finished Evaluation.....')

#######################
#check_set()

#load_mdl = True

train_img(False)

#eval_img(True)




