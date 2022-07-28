[toc]

# 利用拓扑结构识别mnist

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as date
import torchvision
from torch_geometric.nn import GCNConv
device=torch.device('cuda:0')
```

```python
#数据处理
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
```

```python
#加载数据
def data_loader():
    train_loaders=torch.load('./mnist/mnist_train_32.pt')
    test_loaders=torch.load('./mnist/mnist_test_32.pt')
    return train_loaders,test_loaders
```

```pyt
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
#net=Net()#.to(device)
#t=Variable(torch.FloatTensor(np.random.random((16,28,28))))#.to(device)
#t=torch.FloatTensor(np.random.random((16,28,28)))
#net.init_hidden()
#xv=torch.rand(3,2)
#xe=torch.tensor([[0,1],[0,2],[1,2]],dtype=torch.long).T
#xb=torch.tensor([0,0,0])
#a=net(xv,xe,xb)
#a.shape#->torch.Size([32, 28, 10])
```

```py
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

```python
net=Net().to(device)
#batch_size=32
learn_rate=0.001
epoch=20
train_set,test_set=data_loader()
train(net,learn_rate,train_set,test_set,epoch)
```

```python
#结果
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

# 手撸利用余弦识别mnist

```python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as date
import torchvision
#from torch_geometric.nn import GCNConv
#from torch_scatter import scatter_mean
device=torch.device('cuda:0')
```

```python
#处理数据集
def data_loader(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='minist', train=True, transform=transform, download=False)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='minist', train=False, transform=transform, download=False)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loaders,test_loaders
def datatran(a):
    a=a.reshape(-1,28,28)
    x=torch.ones(a.shape[0],a.shape[1],a.shape[2])
    for i in range(1):
        for j in range(a.shape[1]):
            x[:,j:j+1,:]=torch.cosine_similarity(a[:,j,:][:,None,:],a,dim=2)[:,None,:]
    return x
def datacol(data):
    datas=[]
    for x,y in data:
        x=datatran(x)
        datas.append((x,y))
    return datas
train,text=data_loader(32)    
trains=datacol(train)
texts=datacol(text)
torch.save(trains,'./mnist/mnist_train_32t.pt')
torch.save(texts,'./mnist/mnist_test_32t.pt')
```

```python
def data_loader():
    train_loaders=torch.load('./mnist/mnist_train_32t.pt')
    test_loaders=torch.load('./mnist/mnist_test_32t.pt')
    return train_loaders,test_loaders
```

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=nn.Linear(28,32,bias=False)
        self.l2=nn.Linear(32,64,bias=False)
        self.l3=nn.Linear(64,128,bias=False)
        self.l4=nn.Linear(128,256,bias=False)
        self.l5=nn.Linear(256,64)
        self.l6=nn.Linear(64,10)
        self.relu=nn.ReLU()
    def forward(self,x,A):
        x=self.relu(self.l1(torch.matmul(A,x)))
        x=self.relu(self.l2(torch.matmul(A,x)))
        x=self.relu(self.l3(torch.matmul(A,x)))
        x=self.relu(self.l4(torch.matmul(A,x)))
        x=torch.mean(x,dim=1)
        x=self.l5(x)
        x=self.l6(x)
        return x
    def init_hidden(self):
        pass
#net=Net()#.to(device)
#t=Variable(torch.FloatTensor(np.random.random((16,28,28))))#.to(device)
#t=torch.FloatTensor(np.random.random((16,28,28)))
#net.init_hidden()
#A=torch.rand(28,28)
#x=torch.rand(32,28,28)
#a=net(x,A)
#a.shape#->torch.Size([32, 28, 10])
```

```python
def train(model,learn_rate,train_set,test_set,epoch):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate)#Adam
    cost=nn.CrossEntropyLoss()#交叉熵
    A=torch.ones(28,28)
    A=torch.triu(A,diagonal=-1)
    A=torch.tril(A,diagonal=1)
    DSqrt=torch.eye(28,28)*3
    DSqrt[-1,-1]=DSqrt[-1,-1]-1
    DSqrt[0,0]=DSqrt[0,0]-1
    DSqrt=torch.sqrt(DSqrt)
    A=torch.matmul(DSqrt,A)
    A=torch.matmul(A,DSqrt)
    A=Variable(A).to(device)
    for i in range(epoch):
        for x,y in train_set:
            #model.init_hidden()
            #x=Variable(x.reshape(-1,28,28),requires_grad=True).to(device)
            x=Variable(x).to(device)
            y=Variable(y).to(device)
            optimizer.zero_grad()
            out=model(x,A)
            loss=cost(out,y)
            loss.backward()
            optimizer.step()
        print("epoch"+str(i+1)+"：")
        train_acc(model,train_set,A)
        test_acc(model,test_set,A)
    print("Finished training")
```

```python
def train_acc(model,train_set,A):
    correct=0
    total=0
    for datas in train_set:
        x,y=datas
        #model.init_hidden()
        #x=Variable(x.reshape(-1,28,28)).to(device)
        x=Variable(x).to(device)
        y=Variable(y).to(device)
        a=torch.max(model(x,A).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('train_acc:%.2f %%' % (100*correct/total))
def test_acc(model,test_set,A):
    correct=0
    total=0
    for datas in test_set:
        x,y=datas
        #model.init_hidden()
        #x=Variable(x.reshape(-1,28,28)).to(device)
        x=Variable(x).to(device)
        y=Variable(y).to(device)
        a=torch.max(model(x,A).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('test_acc:%.2f %%' % (100*correct/total))
```

```python
net=Net().to(device)
#batch_size=32
learn_rate=0.001
epoch=100
train_set,test_set=data_loader()
train(net,learn_rate,train_set,test_set,epoch)
```

```python
epoch1：
train_acc:83.24 %
test_acc:82.94 %
epoch2：
train_acc:85.58 %
test_acc:84.78 %
epoch3：
train_acc:85.29 %
test_acc:84.95 %
epoch4：
train_acc:86.69 %
test_acc:85.66 %
epoch5：
train_acc:87.22 %
test_acc:86.18 %
epoch6：
train_acc:87.79 %
test_acc:86.78 %
epoch7：
train_acc:87.62 %
test_acc:85.83 %
epoch8：
train_acc:88.68 %
test_acc:87.11 %
epoch9：
train_acc:88.51 %
test_acc:86.16 %
epoch10：
train_acc:88.78 %
test_acc:86.16 %
epoch11：
train_acc:89.02 %
test_acc:86.91 %
epoch12：
train_acc:90.13 %
test_acc:87.56 %
epoch13：
train_acc:89.55 %
test_acc:86.89 %
epoch14：
train_acc:89.43 %
test_acc:86.98 %
epoch15：
train_acc:90.10 %
test_acc:87.51 %
epoch16：
train_acc:89.35 %
test_acc:87.30 %
epoch17：
train_acc:89.51 %
test_acc:86.52 %
epoch18：
train_acc:89.80 %
test_acc:86.79 %
epoch19：
train_acc:88.38 %
test_acc:85.68 %
epoch20：
train_acc:89.41 %
test_acc:86.40 %
epoch21：
train_acc:91.21 %
test_acc:87.76 %
epoch22：
train_acc:90.54 %
test_acc:87.51 %
epoch23：
train_acc:90.56 %
test_acc:87.23 %
epoch24：
train_acc:91.07 %
test_acc:87.52 %
epoch25：
train_acc:91.15 %
test_acc:88.01 %
epoch26：
train_acc:91.10 %
test_acc:87.34 %
epoch27：
train_acc:91.29 %
test_acc:87.84 %
epoch28：
train_acc:90.96 %
test_acc:87.48 %
epoch29：
train_acc:89.89 %
test_acc:86.14 %
epoch30：
train_acc:90.51 %
test_acc:87.28 %
epoch31：
train_acc:90.76 %
test_acc:87.47 %
epoch32：
train_acc:91.92 %
test_acc:87.68 %
epoch33：
train_acc:90.33 %
test_acc:86.58 %
epoch34：
train_acc:92.26 %
test_acc:87.70 %
epoch35：
train_acc:91.18 %
test_acc:87.28 %
epoch36：
train_acc:90.82 %
test_acc:86.88 %
epoch37：
train_acc:92.06 %
test_acc:88.08 %
epoch38：
train_acc:92.74 %
test_acc:87.99 %
epoch39：
train_acc:90.48 %
test_acc:87.03 %
epoch40：
train_acc:91.56 %
test_acc:87.14 %
epoch41：
train_acc:91.92 %
test_acc:87.17 %
epoch42：
train_acc:92.64 %
test_acc:88.04 %
epoch43：
train_acc:92.06 %
test_acc:87.69 %
epoch44：
train_acc:91.44 %
test_acc:86.86 %
epoch45：
train_acc:92.49 %
test_acc:87.11 %
epoch46：
train_acc:92.30 %
test_acc:86.79 %
epoch47：
train_acc:91.29 %
test_acc:86.98 %
epoch48：
train_acc:92.23 %
test_acc:86.99 %
epoch49：
train_acc:92.44 %
test_acc:86.86 %
epoch50：
train_acc:93.12 %
test_acc:87.57 %
epoch51：
train_acc:92.36 %
test_acc:87.36 %
epoch52：
train_acc:92.41 %
test_acc:87.67 %
epoch53：
train_acc:92.54 %
test_acc:87.29 %
epoch54：
train_acc:92.51 %
test_acc:87.17 %
epoch55：
train_acc:92.82 %
test_acc:87.80 %
epoch56：
train_acc:93.22 %
test_acc:87.33 %
epoch57：
train_acc:92.11 %
test_acc:86.65 %
epoch58：
train_acc:92.74 %
test_acc:87.86 %
epoch59：
train_acc:92.84 %
test_acc:87.24 %
epoch60：
train_acc:92.00 %
test_acc:86.66 %
epoch61：
train_acc:91.51 %
test_acc:86.38 %
epoch62：
train_acc:91.90 %
test_acc:86.87 %
epoch63：
train_acc:91.87 %
test_acc:86.83 %
epoch64：
train_acc:92.10 %
test_acc:86.44 %
epoch65：
train_acc:92.93 %
test_acc:87.26 %
epoch66：
train_acc:93.54 %
test_acc:87.75 %
epoch67：
train_acc:92.61 %
test_acc:87.25 %
epoch68：
train_acc:92.34 %
test_acc:86.36 %
epoch69：
train_acc:92.79 %
test_acc:86.80 %
epoch70：
train_acc:92.07 %
test_acc:86.66 %
epoch71：
train_acc:93.24 %
test_acc:87.61 %
epoch72：
train_acc:93.37 %
test_acc:87.56 %
epoch73：
train_acc:92.11 %
test_acc:86.32 %
epoch74：
train_acc:91.97 %
test_acc:87.03 %
epoch75：
train_acc:92.45 %
test_acc:86.63 %
epoch76：
train_acc:93.22 %
test_acc:87.28 %
epoch77：
train_acc:92.83 %
test_acc:87.54 %
epoch78：
train_acc:92.27 %
test_acc:87.11 %
epoch79：
train_acc:93.06 %
test_acc:86.94 %
epoch80：
train_acc:93.83 %
test_acc:87.95 %
epoch81：
train_acc:93.25 %
test_acc:87.03 %
epoch82：
train_acc:92.94 %
test_acc:87.30 %
epoch83：
train_acc:92.84 %
test_acc:87.33 %
epoch84：
train_acc:92.67 %
test_acc:86.76 %
epoch85：
train_acc:93.07 %
test_acc:86.97 %
epoch86：
train_acc:93.50 %
test_acc:87.16 %
epoch87：
train_acc:93.22 %
test_acc:87.03 %
epoch88：
train_acc:93.37 %
test_acc:87.52 %
epoch89：
train_acc:92.58 %
test_acc:86.54 %
epoch90：
train_acc:92.68 %
test_acc:87.07 %
epoch91：
train_acc:91.98 %
test_acc:86.23 %
epoch92：
train_acc:92.91 %
test_acc:86.98 %
epoch93：
train_acc:92.94 %
test_acc:87.78 %
epoch94：
train_acc:92.31 %
test_acc:86.39 %
epoch95：
train_acc:93.54 %
test_acc:87.37 %
epoch96：
train_acc:93.63 %
test_acc:87.13 %
epoch97：
train_acc:92.50 %
test_acc:86.86 %
epoch98：
train_acc:93.14 %
test_acc:87.28 %
epoch99：
train_acc:94.38 %
test_acc:88.10 %
epoch100：
train_acc:93.58 %
test_acc:87.19 %
Finished training
```

