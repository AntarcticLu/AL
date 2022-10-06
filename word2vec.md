[toc]

# skip-gram

[(11条消息) Word2Vec 的pytorch 实现（简单）_我也要做小太阳的博客-CSDN博客_pytorch实现word2vec](https://blog.csdn.net/qq_39215918/article/details/123255248)

```python
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
```

```python
sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like"]
word_sequence = " ".join(sentences).split()
vocab = list(set(word_sequence))
word2idx = {w:i for i,w in enumerate(vocab)}
```

```python
batch_size = 4
embedding_size=2
window = 2
vocab_size = len(vocab)

skip_grams = []
for idx in range(window,len(word_sequence)-window):
    #找到中心词
    center = word2idx[word_sequence[idx]]
    #临近词的索引
    context_idx = list(range(idx-window))+list(range(idx+1,idx+window+1))
    # 找到这些词在word2idx中对应的索引
    context = [word2idx[word_sequence[i]] for i in context_idx]
    for w in context:
        skip_grams.append([center,w])
        
def make_data(skip_grams):
    input_data = []
    output_data = []
    for i in range(len(skip_grams)):
        input_data.append(np.eye(vocab_size)[skip_grams[i][0]])
        output_data.append(skip_grams[i][1])
    return input_data,output_data

input_data,output_data = make_data(skip_grams)
input_data= torch.tensor(input_data,dtype=torch.float32)
output_data = torch.tensor(output_data,dtype=torch.long)
dataset = TensorDataset(input_data, output_data)
train_loader = DataLoader(dataset,batch_size,shuffle =True)
```

```python
class word2vec_(nn.Module):
    def __init__(self):
        super(word2vec_,self).__init__()
        self.w = nn.Parameter(torch.randn(vocab_size,embedding_size).type(torch.float32))
        self.v = nn.Parameter(torch.randn(embedding_size,vocab_size).type(torch.float32))
    def forward(self,x):
#         x:[batch_size,voc_Size]
        
        hidden = torch.matmul(x,self.w)
        #[batch_size,embedding_size]
        
        output = torch.matmul(hidden,self.v )
        return output

model =word2vec_()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

````python
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = 0
    train_num = 0
    for step,(x,y) in enumerate(train_loader):
        x = x
        y = y
        z_hat = model.forward(x)
        loss= criterion(z_hat,y)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        train_loss += loss.item() *len(y)
        train_num+=len(y)
    if (epoch+1) %10==0:
        print(str(epoch+1)+":"+str(loss))
    del x,y,loss,train_loss,train_num
    torch.cuda.empty_cache()
```
10:tensor(2.5852, grad_fn=<NllLossBackward0>)
20:tensor(3.1388, grad_fn=<NllLossBackward0>)
30:tensor(2.9758, grad_fn=<NllLossBackward0>)
40:tensor(4.4472, grad_fn=<NllLossBackward0>)
50:tensor(2.5993, grad_fn=<NllLossBackward0>)
60:tensor(3.6451, grad_fn=<NllLossBackward0>)
70:tensor(3.9716, grad_fn=<NllLossBackward0>)
80:tensor(3.2149, grad_fn=<NllLossBackward0>)
90:tensor(3.6132, grad_fn=<NllLossBackward0>)
100:tensor(1.6699, grad_fn=<NllLossBackward0>)
```
````

```python
for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x,y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
```

# 手写字符集可视化T-SNE

```python
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
import torchvision
import torch.nn as nn
import torch
def data_loader(batch_size):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='../task/minist', train=True, transform=transform, download=False)
    train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='../task/minist', train=False, transform=transform, download=False)
    test_loaders = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loaders,test_loaders
```

```python
train_set,_=data_loader(1000)
tsne = TSNE(n_components=2, init='pca', random_state=0)
x,y=None,None
for x1,y1 in train_set:
    x,y=x1.reshape(x1.shape[0],-1).numpy(),y1.numpy()
    break
X_tsne = tsne.fit_transform(x)

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
 
plot_embedding(X_tsne,y,"t-SNE 2D")
```

