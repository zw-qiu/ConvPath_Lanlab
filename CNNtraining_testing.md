
## 从每个ROI文件夹随机提取训练集(0.7)验证集(0.1)和测试集(0.2)并整合


```python
import os
import glob
import random
import numpy as np
import math
from shutil import copyfile
from scipy import io
from PIL import Image
##读取mat
def LoadInMat(path):
        # the path should also including the name of the .mat file
        File_Disp = io.loadmat(path)
        # Establish an empty list to save the valid name of variables
        NameOfVariable = []
        for key in File_Disp.keys():
            if key == '__globals__' or key == '__version__' or key == '__header__':
                continue
            else:
                NameOfVariable.append(key)
        MatData = {}
        for t in range(len(NameOfVariable)):
            MatData[NameOfVariable[t]] = File_Disp[NameOfVariable[t]].tolist()
        MatData['Keys'] = NameOfVariable
        return MatData
      
path0 = os.getcwd() #获取当前代码文件的路径
path1=path0+'\ROIs'#进入存放ROIs的文件夹
ROIdirs = glob.glob(path1+'\*')#每个ROIs文件夹的路径
numROIdirs=len(ROIdirs)#计算ROIs文件夹的个数#10

for i in range(1,numROIdirs+1):
    temp = glob.glob(path1+'\ROI'+str(i)+'\*')#每个ROIs的路径
    numROIs = len(temp)#计算ROIs的个数(80*80 patch数)
    arr = np.array(range(1,numROIs+1,1))#1--num(ROIs)
    nindex = np.random.permutation(arr)#随机排列
    ntrain = math.floor(numROIs*0.7)
    nvalid = math.floor(numROIs*0.1)
    ntest = numROIs-ntrain-nvalid
    trainind = nindex[0:ntrain]
    validind = nindex[ntrain:(ntrain+nvalid)]
    testind = nindex[(ntrain+nvalid):numROIs]
    ##提取文件到指定文件夹
    labelmat=LoadInMat(path0+'\\Labels\\382412-2_'+str(i)+'.mat')
    label=labelmat['label'][0]#type:list
    labels=[]
    for j in trainind:
        img = np.asarray(Image.open(path1+'\ROI'+str(i)+'\ROI_382412-2_' + str(i) +'_Patch_'+str(j)+'.jpg'))
        Output = {'img': img, 'label': label[j-1]}
        io.savemat(path0+'\\train\\ROI_382412-2_' + str(i) +'_Patch_'+str(j)+'.mat', Output)
    for j in validind:
        img = np.asarray(Image.open(path1+'\ROI'+str(i)+'\ROI_382412-2_' + str(i) +'_Patch_'+str(j)+'.jpg'))
        Output = {'img': img, 'label': label[j-1]}
        io.savemat(path0+'\\valid\\ROI_382412-2_' + str(i) +'_Patch_'+str(j)+'.mat', Output)
    for j in testind:
        img = np.asarray(Image.open(path1+'\ROI'+str(i)+'\ROI_382412-2_' + str(i) +'_Patch_'+str(j)+'.jpg'))
        Output = {'img': img, 'label': label[j-1]}
        io.savemat(path0+'\\test\\ROI_382412-2_' + str(i) +'_Patch_'+str(j)+'.mat', Output)   
        

```

## 读取数据、变换及归一化,构建网络


```python
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob

def LoadInMat(path):
        # the path should also including the name of the .mat file
        File_Disp = io.loadmat(path)
        # Establish an empty list to save the valid name of variables
        NameOfVariable = []
        for key in File_Disp.keys():
            if key == '__globals__' or key == '__version__' or key == '__header__':
                continue
            else:
                NameOfVariable.append(key)
        MatData = {}
        for t in range(len(NameOfVariable)):
            MatData[NameOfVariable[t]] = File_Disp[NameOfVariable[t]].tolist()
        MatData['Keys'] = NameOfVariable
        return MatData
    
def default_loader(path):
    DictMat = LoadInMat(path)
    img = DictMat['img']
    label = DictMat['label'][0][0]-1
    return img, label   

class customData(Dataset):
    def __init__(self, PathofDataset='E:\\Jupyter-notebook\\ConvPath\\train\\*', train=' ',
                 loader=default_loader,transform=''):
        self.listofDataset = glob.glob(PathofDataset)
        self.loader = loader
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.listofDataset)

    def __getitem__(self, item):
        img, label = self.loader(self.listofDataset[item])

        img = np.asarray(img)
        img = img.astype(np.uint8)
        img = transforms.ToPILImage()(img)

#         label = np.asarray(label)
#         img = img/np.max(np.abs(img))
# #         img.astype(np.uint8)
# #         np.swapaxes(img,0,2)
# #         np.swapaxes(img,0,1)
# #         img = Image.fromarray(img)
        img = self.transform(img)
#         print(img.shape())
        
        return img, label
    
     
#旋转和翻转图像
from torchvision import transforms as transforms
transform = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation((-45,45)),#随机旋转
    transforms.ToTensor(),#torchvision的输出是[0,1]的PILImage图像
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
    transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))#归一化到[-0.5,0.5]
])

## Load dataset
#DataLoader为我们提供了对Dataset的读取操作，常用参数有：batch_size(每个batch的大小), 
 #shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)
Train_datasets = customData(PathofDataset='E:\\Jupyter-notebook\\ConvPath\\train1\\*', 
                            train=True, loader=default_loader, transform=transform)
Train_dataloaders = torch.utils.data.DataLoader(Train_datasets,batch_size=10,
                                                shuffle=True, num_workers=0)

Valid_datasets = customData(PathofDataset='E:\\Jupyter-notebook\\ConvPath\\valid1\\*', 
                           train=False, loader=default_loader, transform=transform)
Valid_dataloaders = torch.utils.data.DataLoader(Valid_datasets,batch_size=10,
                                               #shuffle=True, 
                                                num_workers=0)
Test_datasets = customData(PathofDataset='E:\\Jupyter-notebook\\ConvPath\\test1\\*', 
                           train=False, loader=default_loader, transform=transform)
Test_dataloaders = torch.utils.data.DataLoader(Test_datasets,batch_size=10,
                                               #shuffle=True, 
                                                num_workers=0)

classes = ('Tumor cell', 'stroma cell', 'lymphocyte')
#print(Train_dataloaders,Test_dataloaders)


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 20 output channels, 5*5 square convolution
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(20, 20, 3)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(20, 20, 3)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(20* 2 * 2, 20)
        self.fc2 = nn.Linear(20, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 20 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
#         x = F.log_softmax(x,dim=1)
        return x


net = Net()
print(net)

import torch.optim as optim
from torch.optim import lr_scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99995)
```

    Net(
      (conv1): Conv2d(3, 20, kernel_size=(3, 3), stride=(1, 1))
      (pool1): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1))
      (pool2): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
      (conv3): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1))
      (pool3): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=80, out_features=20, bias=True)
      (fc2): Linear(in_features=20, out_features=3, bias=True)
      (dropout): Dropout(p=0.5)
    )


## 训练网络，训练集和验证集准确率


```python
import numpy as np
from torch.autograd import Variable
from scipy import io
loss_save=[]
correct_train=[]
total_train=[]
correct_valid=[]
total_valid=[]
for epoch in range(20):  # 多批次循环
    class_correct_train = list(0. for i in range(3))
    class_total_train = list(0. for i in range(3))
    class_correct_valid = list(0. for i in range(3))
    class_total_valid = list(0. for i in range(3))
    running_loss = 0.0
    count = 0
    for stage in ['train','valid']:
        if stage == 'train':
            dataloader = Train_dataloaders
            net.train(True)
        else:
            dataloader = Valid_dataloaders
            net.train(False)
                
        for data in dataloader:
            count += 1
            #i, data in enumerate(Train_dataloaders,0):
            # 获取输入
            inputs, labels = data
            labels = labels.long()
            #inputs=np.asarray(inputs)
            #labels=np.asarray(labels)
#             inputs = torch.transpose(inputs,1, 3).float()
    #         inputs = Variable(inputs.type(torch.FloatTensor))
    #         labels = Variable(labels.type(torch.FloatTensor))
            #print('inputs')
            #print(torch.max(inputs))
            #print(torch.min(inputs))
            # 梯度置0
            optimizer.zero_grad()
            
            # 正向传播，反向传播，优化
            outputs = net(inputs)
    #         print(outputs.size())
    #         print(labels.size())


            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 打印状态信息
            running_loss += loss.item()
            if count % 100 == 99:    # 每100批次打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                loss_save.append(running_loss)
                #print('running_loss:'+str(loss_save))
                running_loss = 0.0
            
#             
            _, predicted = torch.max(outputs, 1)
#             print(outputs)
#             print(labels)
#             print(inputs)
            c = (predicted == labels).squeeze()
            if stage == 'train':
                for i in range(len(labels.numpy().tolist())):
                #for i in range(10):
                    label = labels[i]
                    class_correct_train[label] += c[i].item()
                    class_total_train[label] += 1
                    #print('class_correct_train: ' + str(class_correct_train) + 'class_total_train: ' + str(class_total_train))
            
            if stage == 'valid':
                for i in range(len(labels.numpy().tolist())):
                #for i in range(10):
                    label = labels[i]
                    class_correct_valid[label] += c[i].item()
                    class_total_valid[label] += 1
                    #print('class_correct_valid: ' + str(class_correct_valid) + 'class_total_valid: ' + str(class_total_valid))
    correct_train.append(class_correct_train)
    correct_valid.append(class_correct_valid)
    total_train.append(class_total_train)
    total_valid.append(class_total_valid)   
    Output = {'loss_save': loss_save,'correct_train':correct_train,'total_train':total_train,'correct_valid':correct_valid,'total_valid':total_valid}
    io.savemat("E:\\Jupyter-notebook\\ConvPath\\result\\loss.mat", Output)
    torch.save(net, "E:\\Jupyter-notebook\\ConvPath\\result\\net_Trained.pth")
print('Finished Training')
```


## 测试集测试及准确率


```python
import numpy as np
from torch.autograd import Variable
from scipy import io

net = Net()
net_dict = net.state_dict()#网络参数的字典
print(net_dict)
classifier = torch.load("E:\\Jupyter-notebook\\ConvPath\\result_2\\net_Trained.pth")
classifier_dict = classifier.state_dict()#提取字典

trained_dict = {k: v for k, v in classifier_dict.items() if k in net_dict}#提取字典里的参数
net_dict.update(trained_dict)
net.load_state_dict(net_dict)

#outputs = net(inputs)

correct_test=[]
total_test=[]
class_correct_test = list(0. for i in range(3))
class_total_test = list(0. for i in range(3))

for data in Test_dataloaders:
    # 获取输入
    inputs, labels = data
    labels = labels.long()
    #输出
    outputs = net(inputs)
          
    _, predicted = torch.max(outputs, 1)
#             print(outputs)
#             print(labels)
#             print(inputs)
    c = (predicted == labels).squeeze()
    for i in range(len(labels.numpy().tolist())):
    #for i in range(10):
        label = labels[i]
        class_correct_test[label] += c[i].item()
        class_total_test[label] += 1
        #print('class_correct_test: ' + str(class_correct_test) + 'class_total_test: ' + str(class_total_test))

correct_test.append(class_correct_test)
total_test.append(class_total_test)   
Output = {'correct_test':correct_test,'total_test':total_test}
io.savemat("E:\\Jupyter-notebook\\ConvPath\\result_2\\test_predict.mat", Output)
print('Finished Testing')


#for i in range(3):
    #print('Accuracy of %5s : %2d %%' % (
        #classes[i], 100 * class_correct[i] / class_total[i]))
```
