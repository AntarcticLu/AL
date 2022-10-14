[toc]

# data

[DataLink](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)

```python
#modelnet40_ply_hdf5_2048
ply_data_test_0_id2file.json
ply_data_test_1_id2file.json
ply_data_test0.h5    #'data' (2048，2048，3)  (-1,1)  'label' (2048,1)   [0,39]
ply_data_test1.h5    #'data'(420，2048，3)   (-1,1)  'label' (420,1)   [0,39]
ply_data_train_0_id2file.json
ply_data_train_1_id2file.json
ply_data_train_2_id2file.json
ply_data_train_3_id2file.json
ply_data_train_4_id2file.json
ply_data_train0.h5   #'data'(2048，2048，3) (-1,1)  'label' (2048,1)   [0,39]
ply_data_train1.h5   #'data'(2048，2048，3) (-1,1)  'label' (2048,1)   [0,39]
ply_data_train2.h5   #'data'(2048，2048，3) (-1,1)  'label' (2048,1)   [0,39]
ply_data_train3.h5   #'data'(2048，2048，3) (-1,1)  'label' (2048,1)   [0,39]
ply_data_train4.h5   #'data'(1648，2048，3) (-1,1)  'label' (1648,1)   [0,39]
shape_names.txt
test_files.txt
train_files.txt
```

![image.png](https://upload-images.jianshu.io/upload_images/28358962-7c5c4db477cc9302.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# cls model

## data.py

```python
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
```

```python
#如果没有下载，就下载数据
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   #获取当前脚本的路径
    DATA_DIR = os.path.join(BASE_DIR, 'data')              #加个data文件夹
    if not os.path.exists(DATA_DIR):                       #如果不存在data文件夹
        os.mkdir(DATA_DIR)                                 #创建data文件夹
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):       #如果不存在数据 
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'  
        zipfile = os.path.basename(www)                                              #返回路径最后文件名
        os.system('wget %s; unzip %s' % (www, zipfile))                              #wget下载 unzip解压
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))                           #移动文件从mode..48到data里
        os.system('rm %s' % (zipfile))                                             #删除mode..48.zip文件
```

```python
def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):         #glob.glob()返回所有匹配到的文件列表
        f = h5py.File(h5_name)   
        data = f['data'][:].astype('float32')  #读取并类型转化 
        label = f['label'][:].astype('int64')  
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)      #tr 9840 te 2468
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
```

```python
def translate_pointcloud(pointcloud): #(1024,3)  转化点云？   μx+b  给分布变了
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])   #[low,high)的均匀分布
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud
```

```python
def normalize_pointcloud(pointcloud):
    center = pointcloud.mean(axis=0)
    pointcloud -= center
    distance = np.linalg.norm(pointcloud, axis=1)
    pointcloud /= distance.max()
    return pointcloud
```

```python
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud
```

```python
# **********Dataset ModelNet40**********

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]    #(1024,3)
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud) 
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
```

## model_cls.py

```python
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)     #(b,1024,1024)
    xx = torch.sum(x**2, dim=1, keepdim=True)         #(b,1,1024)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) #(b,1024,1024)
    #妙啊   -[(xi-xj)^2+(yi-yj)^2+(zi-zj)^2]   
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)  返回top-k的下标
    return idx
```

```python
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)  #b
    num_points = x.size(2)  #1024
    x = x.view(batch_size, -1, num_points)  #(b,3,1024)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points #(b,1,1)

        idx = idx + idx_base   #(b,1024,20)

        idx = idx.view(-1)  #(b*1024*20)
 
    _, num_dims, _ = x.size() #3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]  #根据下表取出前20  (b*1024,20,3)
    feature = feature.view(batch_size, num_points, k, num_dims)   #(b,1024,20,3)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)   #(b,1024,20,3)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()  #(b,6,1024,20)

    return feature, idx  #(b,6,1024,20) (b*1024*20)
```

```python
class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels*in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        # x: (bs, in_channels, num_points, k), y: (bs, feat_channels, num_points, k)
        batch_size, n_dims, num_points, k = x.size()

        y = self.conv0(y) # (bs, out, num_points, k)
        y = self.leaky_relu(self.bn0(y))   #归一化放在了激活函数和卷积之间
        y = self.conv1(y) # (bs, in*out, num_points, k)
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels, self.in_channels) # (bs, num_points, k, out, in)

        x = x.permute(0, 2, 3, 1).unsqueeze(4) # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(y, x).squeeze(4) # (bs, num_points, k, out_channels)
        x = x.permute(0, 3, 1, 2).contiguous() # (bs, out_channels, num_points, k)

        x = self.bn1(x)
        x = self.leaky_relu(x)

        return x
```

```python
class Net(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.adapt_conv1 = AdaptiveConv(6, 64, 6)
        self.adapt_conv2 = AdaptiveConv(6, 64, 64*2)

    def forward(self, x):
        batch_size = x.size(0)#b
        points = x   #(b,3,1024)

        x, idx = get_graph_feature(x, k=self.k)   #(b,6,1024,20)  (b*1024*20)
        p, _ = get_graph_feature(points, k=self.k, idx=idx)  #(b,6,1024,20)
        x = self.adapt_conv1(p, x)             #(b,64,1024,20)
        x1 = x.max(dim=-1, keepdim=False)[0]   #(b,64,1024)

        x, idx = get_graph_feature(x1, k=self.k)  #(b,128,1024,20) (b*1024*20)
        p, _ = get_graph_feature(points, k=self.k, idx=idx)  #(b,6,1024,20)
        x = self.adapt_conv2(p, x)             #(b,64,1024,20)
        x2 = x.max(dim=-1, keepdim=False)[0]   #(b,64,1024) 

        x, _ = get_graph_feature(x2, k=self.k)  #(b,128,1024,20)
        x = self.conv3(x)                       #(b,128,1024,20)
        x3 = x.max(dim=-1, keepdim=False)[0]    #(b,128,1024)

        x, _ = get_graph_feature(x3, k=self.k)  #(b,256,1024,20)
        x = self.conv4(x)                       #(b,256,1024,20)
        x4 = x.max(dim=-1, keepdim=False)[0]    #(b,256,1024)

        x = torch.cat((x1, x2, x3, x4), dim=1)  #(b,512,1024)

        x = self.conv5(x)                       #(b,emb_dim,1024)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  #(b,emb_dim) 无论输入多少，结果为指定大小，2d3d同理
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  #(b,emb_dim)
        x = torch.cat((x1, x2), 1)                             #(b,2*emb_dim)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  #(b,512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  #(b,256)
        x = self.dp2(x)
        x = self.linear3(x)                                              #(b,40)
        return x
```

## train.py

```python
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from importlib import import_module

TRAIN_NAME = __file__.split('.')[0]  #train
```

```python
def parse_arguments():   #秀儿~
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='model_cls', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--Tmax', type=int, default=250, metavar='N',
                        help='Max iteration number of scheduler. ')
    parser.add_argument('--use_sgd', type=int, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=int,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    args = parser.parse_args()

    return args
```

```python
def _init_(args):
    if args.name == '':
        args.name = TRAIN_NAME
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/'+args.name):
        os.makedirs('models/'+args.name)
    if not os.path.exists('models/'+args.name+'/'+'models'):
        os.makedirs('models/'+args.name+'/'+'models')
    os.system('cp {}.py models/{}/{}.py.backup'.format(TRAIN_NAME, args.name, TRAIN_NAME))
    os.system('cp {}.py models/{}/{}.py.backup'.format(args.model, args.name, args.model))
    os.system('cp util.py models' + '/' + args.name + '/' + 'util.py.backup')
    os.system('cp data.py models' + '/' + args.name + '/' + 'data.py.backup')
```

```python
def train(args, io):

    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))
    MODEL = import_module(args.model)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    torch.manual_seed(args.seed)  #设立随机数，方便下次复现     
    if args.gpu_idx < 0:
        io.cprint('Using CPU')
    else:
        io.cprint('Using GPU: {}'.format(args.gpu_idx))
        torch.cuda.manual_seed(args.seed)

    # Load data
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    #Try to load models
    io.cprint('Using model: {}'.format(args.model))
    model = MODEL.Net(args).to(device)  
    print(str(model))

    #model = nn.DataParallel(model)
    #print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.Tmax, eta_min=args.lr)   #余弦退火
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        if epoch < args.Tmax:
            scheduler.step()
        elif epoch == args.Tmax:
            for group in opt.param_groups:
                group['lr'] = 0.0001

        learning_rate = opt.param_groups[0]['lr']
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)#(b,3,1024)
            batch_size = data.size()[0]#b
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint('EPOCH #{}  lr = {}'.format(epoch, learning_rate))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'models/%s/models/model.t7' % args.name)
            io.cprint('Current best saved in: {}'.format('********** models/%s/models/model.t7 **********' % args.name))
```

```python
def test(args, io):
    MODEL = import_module(args.model)
    device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    io.cprint('********** TEST STAGE **********')
    io.cprint('Reload best epoch:')

    #Try to load models
    model = MODEL.Net(args).to(device)
    model.load_state_dict(torch.load('models/%s/models/model.t7' % args.name))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
```

```python
if __name__ == "__main__":
    args = parse_arguments()

    _init_(args)

    io = IOStream('models/' + args.name + '/train.log')
    io.cprint(str(args))

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
```

## until.py

```python
import numpy as np
import torch
import torch.nn.functional as F
```

```python
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
```

```python
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a') # a 是追加模式

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()  # 刷新缓冲区，以防数据丢失。python写数据会先放入缓冲区，若突然关闭，则gg，flush可强制存入，然后清空缓冲区。

    def close(self):
        self.f.close()
```

