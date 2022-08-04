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

![image.png](https://upload-images.jianshu.io/upload_images/28358962-349393835dbdef62.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# CGAN训练mnist

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
        self.l1=nn.Linear(794,512)
        self.l2=nn.Linear(512,256)
        self.l3=nn.Linear(256,128)
        self.l4=nn.Linear(128,64)
        self.l5=nn.Linear(64,1)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
    def forward(self,x,y):
        y=nn.functional.one_hot(y,10)
        x=torch.cat([x,y],dim=1)
        x=self.relu(self.l1(x))
        x=self.relu(self.l2(x))
        x=self.relu(self.l3(x))
        x=self.relu(self.l4(x))
        x=self.sigmoid(self.l5(x))
        return x
```

```py
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.l1=nn.Linear(138,256)
        self.l2=nn.Linear(256,512)
        self.l3=nn.Linear(512,784)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
    def forward(self,x,y):
        y=nn.functional.one_hot(y,10)
        x=torch.cat([x,y],dim=1)
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
        for x,y in train_set:
            bh=x.shape[0]
            x=x.reshape(bh,-1)
            x=x.to(device)
            y=y.to(device)
            rl=torch.ones(bh,1).to(device)
            fl=torch.zeros(bh,1).to(device)
            x=D(x,y)
            dlr=cost(x,rl)
            z=torch.randn(bh,128).to(device)
            ox=D(G(z,y),y)
            dlf=cost(ox,fl)
            dl=dlr+dlf
            doptimizer.zero_grad()
            dl.backward()
            doptimizer.step()
            z=torch.randn(bh,128).to(device)
            ox=D(G(z,y),y)
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
num=torch.tensor([5]).to(device)
out=G(z,num)
out=out.reshape(28,28,1)
plt.imshow(out.cpu().detach().numpy())
```

![img](data:image/png;base64,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)
