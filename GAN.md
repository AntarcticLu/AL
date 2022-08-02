[toc]

[(10条消息) GAN 生成MNIST数据集_小小小小侯的博客-CSDN博客_gan生成数据集](https://blog.csdn.net/weixin_44530058/article/details/119512203)

# GAN训练mnist

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as date
import torchvision
device=torch.device('cuda:0')
```

```python
def data_loader(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='minist', train=True, transform=transform, download=False)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='minist', train=False, transform=transform, download=False)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loaders,test_loaders
```

```python
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.l1=nn.Linear(784,512)
        self.l2=nn.Linear(512,256)
        self.l3=nn.Linear(256,128)
        self.l4=nn.Linear(128,64)
        self.l5=nn.Linear(64,1)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.l1(x))
        x=self.relu(self.l2(x))
        x=self.relu(self.l3(x))
        x=self.relu(self.l4(x))
        x=self.sigmoid(self.l5(x))
        return x
```

```python
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.l1=nn.Linear(128,256)
        self.l2=nn.Linear(256,512)
        self.l3=nn.Linear(512,784)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
    def forward(self,x):
        x=self.relu(self.l1(x))
        x=self.relu(self.l2(x))
        x=self.tanh(self.l3(x))
        return x
```

```python
def train(D,G,learn_rate,train_set,test_set,epoch):
    doptimizer=torch.optim.Adam(D.parameters(),learn_rate)
    goptimizer=torch.optim.Adam(G.parameters(),learn_rate)
    cost=nn.BCELoss()
    for i in range(epoch):
        for x,_ in train_set:
            bh=x.shape[0]
            x=x.reshape(bh,-1)
            x=x.to(device)
            rl=torch.ones(bh,1).to(device)
            fl=torch.zeros(bh,1).to(device)
            x=D(x)
            dlr=cost(x,rl)
            z=torch.randn(bh,128).to(device)
            ox=D(G(z))
            dlf=cost(ox,fl)
            dl=dlr+dlf
            doptimizer.zero_grad()
            dl.backward()
            doptimizer.step()
            z=torch.randn(bh,128).to(device)
            ox=D(G(z))
            gloss=cost(ox,rl)
            goptimizer.zero_grad()
            gloss.backward()
            goptimizer.step()
        print('epoch:'+str(i))
    print("Finished training")
```

```python
G=generator().to(device)
D=discriminator().to(device)
learn_rate=0.001 
epoch=10
train_set,test_set=data_loader(32)
train(D,G,learn_rate,train_set,test_set,epoch)
```

```python
epoch:0
epoch:1
epoch:2
epoch:3
epoch:4
epoch:5
epoch:6
epoch:7
epoch:8
epoch:9
Finished training
```

```python
import cv2 as cv
import matplotlib.pyplot as plt
z=torch.randn(1,128).to(device)
out=G(z)
out=out.reshape(28,28,1)
plt.imshow(out.cpu().detach().numpy())
```

![img](data:image/png;base64,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)