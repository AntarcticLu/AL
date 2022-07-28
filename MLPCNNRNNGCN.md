[toc]

# 绪论

## 本文目的

简单的帮助大家学习深度学习的一些知识。对毕设，研究生都有一定的帮助。

## 相关前提的基础

高等数学、线性代数、概率论与数理统计、python、数据结构

## 主旨

函数拟合

## 小知识

-训练集 测试集 验证集

-拟合 过拟合 欠拟合 

-偏差 方差

# MLP

## 神经元

![一层单个神经元](D:\Typora\image\image-20220628140147090.png)

$a=f(WX+b)$    其中f()为激活函数 

## 激活函数

### 目的

-使其非线性化

### 一般的激活函数

-sigmoid $f(x)=\frac{1}{1+e^{-x}}$  

-Relu $f(x)=\begin{cases}x,x>0\\0,other\end{cases}$  

-tanh $f(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$  

## 简单神经网络的计算

### 损失函数和成本函数

损失函数一般是计算单个样本的损失，成本函数是一批整体的所有样本的损失。

#### 一般的损失函数

$L(y,a)=\frac{1}{2}(a-y)^{2}$  平方差

$L(y,a)=-ylog(a)-(1-y)log(1-a)$   交叉熵（简单推导）

#### 成本函数

$J(W,b)=\frac{1}{m}\sum^{m}_{i=1}L(y^{i},a^{i})$

### 超参数

超参数一般是指调参侠自己调节的参数，这些参数没有标准，根据网络情况而定。

-学习率

-迭代次数

### 正向传播

![image-20220628141551014](D:\Typora\image\image-20220628141551014.png)

#### 假设

-激活函数$f(x)=sigmoid(x)$

-损失函数$L(y,a)=\frac{1}{2}(a-y)^{2}$

-成本函数$J(W,b)=\frac{1}{m}\sum^{m}_{i=1}L(y^{i},a^{i})$

#### 计算

$a=f(WX+b)$ 

$L(y,a)=\frac{1}{2}(a-y)^{2}$

$J(W,b)=\frac{1}{m}\sum^{m}_{i=1}L(y^{i},a^{i})$

### 反向传播

#### 梯度下降公式

$W:=W-\alpha dW$  , $dW=\frac{\partial J}{\partial W}$

$b:=b-\alpha db$ , $db=\frac{\partial J}{\partial b}$

#### 计算

$\frac{\partial J}{\partial W}=\frac{1}{m}\sum^{m}_{i=1}(a^{i}-y^{i})a^{i}(1-a^{i})x^{i}=\frac{1}{m}(a-y) \times a(1-a) * X^{T}$

$\frac{\partial J}{\partial b}=\frac{1}{m}\sum^{m}_{i=1}(a^{i}-y^{i})a^{i}(1-a^{i})$

#### 一些问题

-梯度消失和梯度爆炸

-归一化

-正则化

-优化器

## 实例一

![image-20220628154952060](D:\Typora\image\image-20220628154952060.png)

训练一个单层单神经元的简单神经网络模拟 and 过程。

激活函数使用Relu，损失函数使用平方差，训练10次，学习率为0.1

### 代码

```python
import numpy as np
x=np.array([[1,1],[0,0],[1,0],[0,1]])
y=np.array([1,0,0,0]).reshape(1,x.shape[0])  #（4，2）
def predict(x,w,b):
    return (np.dot(w,x.T)+b)>0
def train(x,y,epochs,rate):
    w=np.zeros(x.shape[1]).reshape(1,x.shape[1])
    b=0
    cost=[]
    for i in range(epochs):
        yp=predict(x,w,b)
        cost.append((1/(2*x.shape[0]))*np.sum((yp-y)*(yp-y)))
        w+=(1/x.shape[0])*np.dot(y-yp,x)*rate
        b+=(1/x.shape[0])*rate*np.sum(y-yp)
    return w,b,cost
we,be,cost=train(x,y,10,0.1)
```

```python
print(we,be)
print(predict(np.array([0,0]),we,be))
print(predict(np.array([1,0]),we,be))
print(predict(np.array([0,1]),we,be))
print(predict(np.array([1,1]),we,be))
print(cost)

#运行结果
[[0.025 0.025]] -0.02500000000000001
[False]
[False]
[False]
[ True]
[0.125, 0.375, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

```python
import matplotlib.pyplot as plt
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('learning rate=0.1')
plt.plot(cost)
plt.show()
```

![image-20220628155605585](D:\Typora\image\image-20220628155605585.png)

## 复杂神经网络的计算

![image-20220628164008154](D:\Typora\image\image-20220628164008154.png)

### 正向传播 

$a_{1}=Relu(W_{1}X+b_{1})$ 

$a_{2}=Relu(W_{2}a_{1}+b_{2})$ 

$L(y,a_{2})=\frac{1}{2}(a_{2}-y)^{2}$

$J(W_{1},b_{1},W_{2},b_{2})=\frac{1}{m}\sum^{m}_{i=1}L(y^{i},a^{i}_{2})$

### 反向传播

$\frac{\partial J}{\partial W_{2}}=\frac{1}{m}(a_{2}-y)*a_{1}^{T}$

$\frac{\partial J}{\partial b_{2}}=\frac{1}{m}\sum^{m}_{i=1}(a^{i}_{2}-y^{i})$

$\frac{\partial J}{\partial W_{1}}=\frac{1}{m}W^{T}_{2}*(a_{2}-y)*X^{T}$

$\frac{\partial J}{\partial b_{1}}=\frac{1}{m}\sum^{m}_{i=1}(a^{i}_{2}-y^{i})W^{i}_{2}$

## 实例二

mnist手写字符集，开源字符集，训练集6W张，测试集1W张，每张图片大小(1,28,28)

利用全连接神经网络实现字符的判断

[MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges](http://yann.lecun.com/exdb/mnist/)

要求：

输入层 784 隐藏层 300 100 输出层 10

激活函数 sigmoid 损失函数$L(y,a)=\frac{1}{2}(a-y)^{2}$

### 代码

```python
import numpy as np
import matplotlib.pyplot as plt
import math
```

```python
def image_load(ti,count):
    image_ti=[]
    for i in range(count):
        image_ti.append([])
        for j in range(i*784+16,(i+1)*784+16):
            image_ti[i].append(ti[j])
    return image_ti
def label_load(tl,count):
    label_tl=[]
    for i in range(count):
        labels=[0.1]*10
        labels[tl[i+8]]=0.9
        label_tl.append(labels)
    return label_tl
def get_dataset(count_tr,count_te):
    f=open('./mnist/train-images.idx3-ubyte','rb')
    tri=f.read()
    f=open('./mnist/train-labels.idx1-ubyte','rb')
    trl=f.read()
    f=open('./mnist/t10k-images.idx3-ubyte','rb')
    tei=f.read()
    f=open('./mnist/t10k-labels.idx1-ubyte','rb')
    tel=f.read()
    f.close
    image_tri=image_load(tri,count_tr)
    image_tei=image_load(tei,count_te)
    train_images=np.array(image_tri).T
    test_images=np.array(image_tei).T
    label_trl=label_load(trl,count_tr)
    label_tel=label_load(tel,count_te)
    train_labels=np.array(label_trl).T
    test_labels=np.array(label_tel).T
    return train_images,train_labels,test_images,test_labels
train_images,train_labels,test_images,test_labels=get_dataset(60000,10000)
```

```python
def evaluation(paramater,x,y):
    v,c=postive_propagation(x,y,paramater,layer_size)
    vs=list(np.argmax(v['a'+str(layer_size)],axis=0)-np.argmax(y,axis=0))
    return vs.count(0)/len(vs)
def predict(x,w,b):
    #return np.maximum(0,np.dot(w,x)+b)
    return 1/(1+np.exp(-(np.dot(w,x)+b)))
def postive_propagation(mx,my,paramater,layer_size):
    variable={}
    variable['a0']=mx
    for i in range(layer_size):
        variable['a'+str(i+1)]=predict(variable['a'+str(i)],paramater['w'+str(i+1)],paramater['b'+str(i+1)])
    delta=my-variable['a'+str(layer_size)]
    #delta=np.where(np.logical_and(delta>0,delta<2),1,0)
    cost=(1/(2*mx.shape[1]))*np.sum(delta*delta)
    return variable,cost
def negative_propagation(my,paramater,layer_size,variable,rate):
    delta=my-variable['a'+str(layer_size)]
    #delta=np.where(np.logical_and(delta>0,delta<2),1,0)
    for i in range(layer_size-1,-1,-1):
        diff=variable['a'+str(i+1)]*(1-variable['a'+str(i+1)])
        paramater['w'+str(i+1)]+=(1/(my.shape[1]))*np.dot(delta*diff,variable['a'+str(i)].T)*rate
        paramater['b'+str(i+1)]+=(1/(my.shape[1]))*np.sum(delta*diff,axis=1).reshape(paramater['b'+str(i+1)].shape[0],1)*rate
        delta=np.dot(paramater['w'+str(i+1)].T,delta*diff)
def train(x,y,epochs,rate,batch_size,net_size):
    paramater={}
    costs=[]
    accuracy_rate=[]
    for i in range(layer_size):
        paramater['w'+str(i+1)]=np.random.uniform(-0.1,0.1,(net_size[i+1],net_size[i]))
        paramater['b'+str(i+1)]=np.zeros((net_size[i+1],1))
    for i in range(epochs):
        k=math.ceil(x.shape[1]/batch_size)
        for j in range(k):
            if j==k-1:
                mx=x[:,j*batch_size:]
                my=y[:,j*batch_size:]
            else:
                mx=x[:,j*batch_size:(j+1)*batch_size]
                my=y[:,j*batch_size:(j+1)*batch_size]
            variable,cost=postive_propagation(mx,my,paramater,layer_size)
            costs.append(cost)
            negative_propagation(my,paramater,layer_size,variable,rate)
        accuracy=evaluation(paramater,test_images,test_labels)
        print(i,accuracy)
        if len(accuracy_rate)>0 and accuracy<accuracy_rate[-1]:
            return paramater_old,costs,accuracy_rate
        accuracy_rate.append(accuracy)
        paramater_old=paramater
    return paramater,costs,accuracy_rate
```

```python
net_size=[784,300,100,10]
layer_size=len(net_size)-1
paramater,costs,accuracy_rate=train(train_images,train_labels,200,0.01,1,net_size)

#结果
0 0.8936
1 0.898
2 0.9136
3 0.9093
```

```python
np.save('paramater2.npy',paramater) #存储模型参数
```

```python
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('paramater2.npy_rate=0.01')
plt.plot(accuracy_rate)
plt.show()
```

![image-20220628171500301](D:\Typora\image\image-20220628171500301.png)

# CNN

## Conv

原因：图片有时只需要部分区域特征

![image-20220628235552887](D:\Typora\image\image-20220628235552887.png)

## Pooling

-maxpooling

-sumpooling

-meanpooling

## Padding

原因：有时需要边缘的特征

![image-20220629000500936](D:\Typora\image\image-20220629000500936.png)

## LeNet-5

![image-20220629000616032](D:\Typora\image\image-20220629000616032.png)

$h:=\frac{h+2p-f}{s}$+1

fitler(channel,h,w,stride,padding)

one layer: input(m,1,32,32)--fitler(6,5,5,1,0)-->C1(m,6,28,28)--fitler(6,2,2,2,0)-->S2(m,6,14,14)

two layer: S2(m,6,14,14)--fitler(16,5,5,1,0)-->C3(m,16,10,10)--fitler(16,2,2,2,0)-->S4(m,16,5,5)

three layer: S4(m,16,5,5)--(m,400)-->C5(m,120)

four layer: C5(m,120)-->F6(m,84)

five layer: F6(m,84)-->output(m,10)

## 实例

利用LeNet-5模型训练mnist数据集

建议使用pytorch，jupyter notebook

### 代码

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as date
import torchvision

def data_loader(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='minist', train=True, transform=transform, download=False)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='minist', train=False, transform=transform, download=False)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loaders,test_loaders
```

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.cp1=nn.Sequential(nn.Conv2d(1,6,(3,3),1,1),nn.ReLU(),nn.AvgPool2d(2))
        self.cp2=nn.Sequential(nn.Conv2d(6,16,(5,5),1,0),nn.ReLU(),nn.AvgPool2d(2))
        self.fc1=nn.Sequential(nn.Linear(400,200),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(200,100),nn.ReLU())
        self.out=nn.Sequential(nn.Linear(100,10))
    def forward(self,x):
        x=self.cp1(x)
        x=self.cp2(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.out(x)
        return x

def train(model,learn_rate,train_set,test_set,epoch):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate)#Adam
    cost=nn.CrossEntropyLoss()#交叉熵
    for i in range(epoch):
        running_loss=0
        for j,(x,y) in enumerate(train_set):
            x=Variable(x,requires_grad=True)
            y=Variable(y)
            optimizer.zero_grad()
            out=model(x)
            loss=cost(out,y)
            loss.backward()
            optimizer.step()
            running_loss+=loss
            '''
            if (j+1)%200==0:
                print('[%d,%5d] loss: %.3f' % (i+1,j+1,running_loss/200))
                running_loss=0
            '''
        print("epoch"+str(i+1)+"：")
        train_acc(model,train_set)
        test_acc(model,test_set)
    print("Finished training")
```

```python
def train_acc(model,train_set):
    correct=0
    total=0
    for datas in train_set:
        x,y=datas
        x=Variable(x)
        a=torch.max(model(x).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('train_acc:%.2f %%' % (100*correct/total))
def test_acc(model,test_set):
    correct=0
    total=0
    for datas in test_set:
        x,y=datas
        x=Variable(x)
        a=torch.max(model(x).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('test_acc:%.2f %%' % (100*correct/total))

```

```python
net=Net()
batch_size=32
learn_rate=0.001
epoch=50
train_set,test_set=data_loader(batch_size)
train(net,learn_rate,train_set,test_set,epoch)
```

![image-20220629080407606](D:\Typora\image\image-20220629080407606.png)

# RNN

原因：处理序列模型

特点：共享权重

## 单层单向RNN

![image-20220629070632532](D:\Typora\image\image-20220629070632532.png)

$a^{<i-1>}=f(W_{x}X^{<i-1>}+W_{a}a^{<i-2>}+b_{a})$

$\hat{y}^{<i-1>}=g(W_{y}a^{<i-1>}+b_{y})$

$a^{<i>}=f(W_{x}X^{<i>}+W_{a}a^{<i-1>}+b_{a})$

$\hat{y}^{<i>}=g(W_{y}a^{<i>}+b_{y})$

$a^{<i+1>}=f(W_{x}X^{<i+1>}+W_{a}a^{<i>}+b_{a})$

$\hat{y}^{<i+1>}=g(W_{y}a^{<i+1>}+b_{y})$

## 单层双向RNN

![image-20220629072344838](D:\Typora\image\image-20220629072344838.png)

## 深度RNN

![image-20220629071534116](D:\Typora\image\image-20220629071534116.png)

## 另一种分类

一对多、多对一、多对多

## 一些问题

-由于网络过于长而导致梯度消失

## 实例

利用RNN模型实现识别mnist

### 代码

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as date
import torchvision
def data_loader(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='minist', train=True, transform=transform, download=False)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='minist', train=False, transform=transform, download=False)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
return train_loaders,test_loaders
```

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.r1=nn.RNNCell(28,64)
        self.r2=nn.RNNCell(64,128)
        self.r3=nn.RNNCell(128,64)
        self.fc=nn.Linear(64,10)
    def forward(self,x):
        for i in range(x.shape[1]):
            x0=x[:,i,:].reshape(16,28)
            self.h1=self.r1(x0,self.h1)
            self.h2=self.r2(self.h1,self.h2)
            self.h3=self.r3(self.h2,self.h3)
        x=self.fc(self.h3)
        return x
    def init_hidden(self):
        self.h1=torch.randn(16,64)#.to(device)  #(num_layers,batch_size,hidden_size)
        self.h2=torch.randn(16,128)
        self.h3=torch.randn(16,64)
def train(model,learn_rate,train_set,test_set,epoch):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate)#Adam
    cost=nn.CrossEntropyLoss()#交叉熵
    for i in range(epoch):
        running_loss=0
        for j,(x,y) in enumerate(train_set):
            model.init_hidden()
            x=Variable(x.reshape(-1,28,28),requires_grad=True)
            y=Variable(y)
            optimizer.zero_grad()
            #x,y=x.to(device),y.to(device)
            out=model(x)
            loss=cost(out,y)
            loss.backward()
            optimizer.step()
            running_loss+=loss
        print("epoch"+str(i+1)+"：")
        train_acc(model,train_set)
        test_acc(model,test_set)
print("Finished training")
```

```python
def train_acc(model,train_set):
    correct=0
    total=0
    for datas in train_set:
        x,y=datas
        model.init_hidden()
        x=Variable(x.reshape(-1,28,28))#.to(device)
        y=y#.to(device)
        a=torch.max(model(x).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('train_acc:%.2f %%' % (100*correct/total))
def test_acc(model,test_set):
    correct=0
    total=0
    for datas in test_set:
        model.init_hidden()
        x,y=datas
        x=Variable(x.reshape(-1,28,28))#.to(device)
        y=y#.to(device)
        a=torch.max(model(x).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('test_acc:%.2f %%' % (100*correct/total))
```

```python
net=Net()#.to(device)
batch_size=16
learn_rate=0.001
epoch=10
train_set,test_set=data_loader(batch_size)
train(net,learn_rate,train_set,test_set,epoch)
```

![image-20220629080721704](D:\Typora\image\image-20220629080721704.png)

# GCN

## 理论

[A Gentle Introduction to Graph Neural Networks (distill.pub)](https://distill.pub/2021/gnn-intro/)

## 实例

利用GCN模型训练mnist

### 代码

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as date
import torchvision
from torch_geometric.nn import GCNConv

def data_loader(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='minist', train=True, transform=transform, download=False)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='minist', train=False, transform=transform, download=False)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loaders,test_loaders
def datatran(a):
    a=a.reshape(-1,28,28)
    va=[]
    ea=[]
    vam=-1
    ba=[]
    for i in range(a.shape[0]):
        ba+=[i]*(a[i]>0).sum()
        va.append((a[i]>0).nonzero())
        temp1=[-1]*a.shape[2]
        temp2=[-1]*a.shape[2]
        for ij in range(a.shape[1]):
            for jj in range(a.shape[2]):
                if a[i][ij][jj]>0:
                    vam+=1
                    temp2[jj]=vam
                    if ij-1>0 and jj-1>0 and a[i][ij-1][jj-1]>0:
                        ea.append([temp1[jj-1],vam])
                    if ij-1>0 and a[i][ij-1][jj]>0:
                        ea.append([temp1[jj],vam])
                    if ij-1>0 and jj+1<a.shape[2] and a[i][ij-1][jj+1]>0:
                        ea.append([temp1[jj+1],vam])
                    if ij-1>0 and a[i][ij][jj-1]>0:
                        ea.append([temp2[jj-1],vam])
            temp1=temp2
    va=torch.cat(va,dim=0)
    va=va.type(torch.float)
    ea=torch.tensor(ea,dtype=torch.long)
    ba=torch.tensor(ba)
    return va,ea,ba
def datacol(data):
    datas=[]
    for x,y in data:
        va,ea,bc=datatran(x)
        datas.append((va,ea,bc,y))
    return datas
train,text=data_loader(32)    
trains=datacol(train)
texts=datacol(text)

torch.save(trains,'./mnist/mnist_train_32.pt')
torch.save(texts,'./mnist/mnist_test_32.pt')
```

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as date
import torchvision
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
device=torch.device('cuda:0')

def data_loader():
    train_loaders=torch.load('./mnist/mnist_train_32.pt')
    test_loaders=torch.load('./mnist/mnist_test_32.pt')
    return train_loaders,test_loaders
```

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.g1=GCNConv(2,8)
        self.g2=GCNConv(8,32)
        self.g3=GCNConv(32,64)
        self.g4=GCNConv(64,128)
        self.g5=GCNConv(128,256)
        self.f6=nn.Linear(256,64)
        self.f7=nn.Linear(64,10)
        self.relu=nn.ReLU()
    def forward(self,x,e,b):
        x=self.relu(self.g1(x,e))
        x=self.relu(self.g2(x,e))
        x=self.relu(self.g3(x,e))
        x=self.relu(self.g4(x,e))
        x=self.relu(self.g5(x,e))
        x=scatter_mean(x,b,dim=0)
        x=self.relu(self.f6(x))
        x=self.f7(x)
        return x
    def init_hidden(self):
        pass
    
def train(model,learn_rate,train_set,test_set,epoch):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate)#Adam
    cost=nn.CrossEntropyLoss()#交叉熵
    for i in range(epoch):
        running_loss=0
        for xv,xe,xb,y in train_set:
            #model.init_hidden()
            #x=Variable(x.reshape(-1,28,28),requires_grad=True).to(device)
            xv=Variable(xv).to(device)
            xe=Variable(xe.T).to(device)
            xb=Variable(xb).to(device)
            y=Variable(y).to(device)
            optimizer.zero_grad()
            #x,y=x.to(device),y.to(device)
            out=model(xv,xe,xb)
            loss=cost(out,y)
            loss.backward()
            optimizer.step()
            running_loss+=loss
        print("epoch"+str(i+1)+"：")
        train_acc(model,train_set)
        test_acc(model,test_set)
    print("Finished training")
```

```python
def train_acc(model,train_set):
    correct=0
    total=0
    for datas in train_set:
        xv,xe,xb,y=datas
        #model.init_hidden()
        #x=Variable(x.reshape(-1,28,28)).to(device)
        xv=Variable(xv).to(device)
        xe=Variable(xe.T).to(device)
        xb=Variable(xb).to(device)
        y=y.to(device)
        a=torch.max(model(xv,xe,xb).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('train_acc:%.2f %%' % (100*correct/total))
def test_acc(model,test_set):
    correct=0
    total=0
    for datas in test_set:
        xv,xe,xb,y=datas
        #model.init_hidden()
        #x=Variable(x.reshape(-1,28,28)).to(device)
        xv=Variable(xv).to(device)
        xe=Variable(xe.T).to(device)
        xb=Variable(xb).to(device)
        y=y.to(device)
        a=torch.max(model(xv,xe,xb).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('test_acc:%.2f %%' % (100*correct/total))
```

```py
net=Net().to(device)
#batch_size=32
learn_rate=0.001
epoch=20
train_set,test_set=data_loader()
train(net,learn_rate,train_set,test_set,epoch)

#运行结果
epoch1：
train_acc:55.34 %
test_acc:54.79 %
epoch2：
train_acc:74.35 %
test_acc:75.43 %
epoch3：
train_acc:85.19 %
test_acc:85.62 %
epoch4：
train_acc:87.95 %
test_acc:88.41 %
epoch5：
train_acc:89.01 %
test_acc:89.35 %
epoch6：
train_acc:89.76 %
test_acc:89.61 %
epoch7：
train_acc:90.49 %
test_acc:90.40 %
epoch8：
train_acc:90.98 %
test_acc:91.10 %
epoch9：
train_acc:92.00 %
test_acc:92.55 %
epoch10：
train_acc:92.43 %
test_acc:92.79 %
epoch11：
train_acc:92.67 %
test_acc:92.98 %
epoch12：
train_acc:92.64 %
test_acc:92.98 %
epoch13：
train_acc:92.99 %
test_acc:93.46 %
epoch14：
train_acc:93.33 %
test_acc:93.58 %
epoch15：
train_acc:93.40 %
test_acc:93.76 %
epoch16：
train_acc:93.46 %
test_acc:93.72 %
epoch17：
train_acc:93.47 %
test_acc:93.60 %
epoch18：
train_acc:92.86 %
test_acc:92.96 %
epoch19：
train_acc:93.53 %
test_acc:93.84 %
epoch20：
train_acc:93.32 %
test_acc:93.37 %
Finished training
```

