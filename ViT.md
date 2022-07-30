[toc]

[(10条消息) Vision Transformer详解（附代码）_鬼道2022的博客-CSDN博客_视觉transformer代码](https://blog.csdn.net/qq_38406029/article/details/122157116)

# 手撸ViT含多头注意力

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
def data_loader(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='minist', train=True, transform=transform, download=False)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='minist', train=False, transform=transform, download=False)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loaders,test_loaders
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self,vector,head):
        super(MultiHeadAttention,self).__init__()
        self.vector=vector
        self.head=head
        self.heads=vector//head
        self.lq=nn.Linear(self.heads,self.heads,bias=False)
        self.lk=nn.Linear(self.heads,self.heads,bias=False)
        self.lv=nn.Linear(self.heads,self.heads,bias=False)
        self.lo=nn.Linear(self.head*self.heads,self.vector)
    def forward(self,Q,K,V):
        batch=Q.shape[0]
        ql,kl,vl=Q.shape[1],K.shape[1],V.shape[1]
        Q=Q.reshape(batch,ql,self.head,self.heads)
        K=K.reshape(batch,kl,self.head,self.heads)
        V=V.reshape(batch,vl,self.head,self.heads)
        Q,K,V=self.lq(Q),self.lk(K),self.lv(V)
        Q,K,V=Q.transpose(1,2),K.transpose(1,2).transpose(2,3),V.transpose(1,2)
        out=torch.matmul(Q,K)
        out=torch.softmax(out/(self.vector**(1/2)),dim=3)
        out=torch.matmul(out,V).transpose(1,2).reshape(batch,ql,-1)
        out=self.lo(out)
        return out
#net=MultiHeadAttention(10,2)
#q=torch.rand(2,5,10)
#k=torch.rand(2,6,10)
#v=torch.rand(2,6,10)
#a=net(q,k,v)
#a.shape#torch.Size([2, 5, 10])
```

```python
class TEncoder(nn.Module):
    def __init__(self,vector,head,dropout,ff):
        super(TEncoder,self).__init__()
        self.MHA=MultiHeadAttention(vector,head)
        self.norm=nn.LayerNorm(vector)
        self.l1=nn.Linear(vector,ff)
        self.l2=nn.Linear(ff,vector)
        self.relu=nn.ReLU()
        self.do=nn.Dropout(dropout)
    def forward(self,x):
        x=self.MHA(self.norm(x),self.norm(x),self.norm(x))+x
        x=self.do(x)
        x=self.l2(self.relu(self.l1(self.norm(x))))+x
        x=self.do(x)
        return x
#net=TEncoder(10,2,0,20)
#q=torch.rand(2,5,10)
#a=net(q)
#a.shape#torch.Size([2, 5, 10])
```

```python
class Net(nn.Module):
    def __init__(self,batch,xsize,vector1,vector2,head,dropout,ff):
        super(Net,self).__init__()
        self.vector1=vector1
        self.lx=int(vector1**(1/2))
        self.lenlx=int(xsize//self.lx)
        self.lenx=int(self.lenlx**2)
        self.posem=nn.Parameter(torch.randn(batch,self.lenx+1,vector2))
        self.clas=nn.Parameter(torch.randn(batch,1,vector2))
        self.tencoder=TEncoder(vector2,head,dropout,ff)
        self.l1=nn.Linear(vector1,vector2,bias=False)
        self.l2=nn.Linear(vector2,vector2//2)
        self.l3=nn.Linear(vector2//2,10)
        self.relu=nn.ReLU()
    def forward(self,x,layer):
        batch=x.shape[0]
        x=x.reshape(-1,28,28)
        x=torch.cat([x[:,i*self.lx:(i+1)*self.lx,j*self.lx:(j+1)*self.lx].reshape(1,-1) for i in range(self.lenlx) for j in range(self.lenlx)],dim=0)
        x=torch.cat([x[:,i*(self.vector1):(i+1)*self.vector1] for i in range(batch)],dim=0)
        x=x.reshape(batch,self.lenx,self.vector1)
        x=self.l1(x)
        x=torch.cat([x,self.clas],dim=1)
        x+=self.clas
        for i in range(layer):
            x=self.tencoder(x)
        x=torch.mean(x,dim=1)
        x=self.relu(self.l2(x))
        x=self.l3(x)
        return x
#net=Net(32,28,16,32,2,0,64)#.to(device)
#A=torch.rand(28,28)
#x=torch.rand(32,1,28,28)
#a=net(x,2)
#a.shape#->torch.Size([32, 28, 10])
```

```python
def train(model,learn_rate,train_set,test_set,epoch,layer):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate)#Adam
    cost=nn.CrossEntropyLoss()#交叉熵
    for i in range(epoch):
        for x,y in train_set:
            x=Variable(x).to(device)
            y=Variable(y).to(device)
            optimizer.zero_grad()
            out=model(x,layer)
            loss=cost(out,y)
            loss.backward()
            optimizer.step()
        print("epoch"+str(i+1)+"：")
        tacc(model,train_set,1)
        tacc(model,test_set,0)
    print("Finished training")
```

```py
def tacc(model,tset,string):
    correct=0
    total=0
    st={1:'train_acc',0:'test_acc'}
    for datas in tset:
        x,y=datas
        x=Variable(x).to(device)
        y=Variable(y).to(device)
        a=torch.max(model(x,layer).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print(st[string]+':'+str((100*correct/total).item()))
```

```python
net=Net(16,28,49,64,1,0.1,128).to(device)
learn_rate=0.001
epoch=10
layer=1
train_set,test_set=data_loader(16)
train(net,learn_rate,train_set,test_set,epoch,layer)
```

```python
epoch1：
train_acc:59.88833236694336
test_acc:59.5099983215332
epoch2：
train_acc:70.07666778564453
test_acc:70.58999633789062
epoch3：
train_acc:76.82333374023438
test_acc:76.77999877929688
epoch4：
train_acc:80.70000457763672
test_acc:80.54999542236328
epoch5：
train_acc:82.50833129882812
test_acc:82.47000122070312
epoch6：
train_acc:85.32833099365234
test_acc:84.81999969482422
epoch7：
train_acc:85.87833404541016
test_acc:85.43999481201172
epoch8：
train_acc:86.48999786376953
test_acc:85.77999877929688
epoch9：
train_acc:87.03166961669922
test_acc:86.7699966430664
epoch10：
train_acc:88.63666534423828
test_acc:88.32999420166016
Finished training
```

