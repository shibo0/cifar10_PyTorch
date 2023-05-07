import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import *

#超参数定义
EPOCH = 15  # 训练轮数
BATCH_SIZE = 64  # 训练验证批次
LR = 0.001  # 学习率

#数据归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

#训练集和验证集加载
print('==> Loading dataset..')
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            normalize,  #数据归一化
        ]), download=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True)
        
val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,  #数据归一化
        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True)

# Model
print('==> Building model..')
#model = VGG('VGG19')
model = ResNet18()
#model = PreActResNet18()
#model = GoogLeNet()
#model = DenseNet121()
#model = ResNeXt29_2x64d()
#model = MobileNet()
#model = MobileNetV2()
#model = DPN92()
#model = ShuffleNetG2()
#model = SENet18()
#model = ShuffleNetV2(1)
#model = EfficientNetB0()
#model = RegNetX_200MF()
#model = SimpleDLA()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')	#设置GPU
model  = model.to(device)	#将模型参数传入GPU

#定义损失函数，分类任务常用交叉信息熵损失函数
criterion = nn.CrossEntropyLoss()
#torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
optimizer = optim.Adam(model.parameters(),lr=LR)

#模型训练
print('==> Starting train model..')
for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (inputs,labels) in enumerate(train_loader):
        optimizer.zero_grad()  #清空上一轮的梯度
        inputs,labels = inputs.to(device),labels.to(device)  #数据及标签均送入GPU或CPU
        
        outputs = model(inputs)  #将数据inputs传入模型中，前向传播    
        loss = criterion(outputs,labels)   #计算损失函数        
                
        loss.backward()  #反向传播
        optimizer.step()  #参数更新
        
        train_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if  i % 100 == 0:
            print('Epoch{} Iter{} | Loss:{:.4f}, Accuracy:{:.4f}'.format(epoch, i, train_loss/total, correct/total))

    print('Epoch{} | Loss:{:.4f}, Accuracy:{:.4f}\n'.format(epoch, train_loss/total, correct/total))


#保存模型参数
torch.save(model,'ResNet18_cifar10.pt')
print('ResNet18_cifar10.pt saved')


#测试
print('==> Starting val model..')
model.eval()
val_loss = 0
correct = 0
total = 0
for i, (inputs,labels) in enumerate(val_loader):
    inputs,labels = inputs.to(device),labels.to(device)

    outputs = model(inputs)
    loss = criterion(outputs,labels)   #计算损失函数     
    
    val_loss += loss.item()
    _, predicted = outputs.max(dim=1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
    
print('Val Loss:{:.4f}, Accuracy:{:.4f}'.format(val_loss/total, correct/total))


