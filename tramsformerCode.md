[toc]

# 手撕transformer

出处：

[(10条消息) 【手撕Transformer】Transformer输入输出细节以及代码实现（pytorch）_顾道长生'的博客-CSDN博客_transformer输入输出](https://blog.csdn.net/wl1780852311/article/details/121033915)

```python
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

#自制数据集
             # Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],      # E: 结束符号
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]             # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0


src_vocab = {'P':0, '我':1, '是':2, '学':3, '生':4, '喜':5, '欢':6,'习':7,'男':8}   # 词源字典  字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)                 # 字典字的个数

tgt_vocab = {'S':0, 'E':1, 'P':2, 'I':3, 'am':4, 'a':5, 'student':6, 'like':7, 'learning':8, 'boy':9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}                               # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_vocab)                                                     # 目标字典尺寸

src_len = len(sentences[0][0].split(" "))                                           # Encoder输入的最大长度 5
tgt_len = len(sentences[0][1].split(" "))                                           # Decoder输入输出最大长度 5
src_len,tgt_len
```

```py
# 把sentences 转换成字典索引
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
      enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] 
      dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] 
      dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] 
      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
print(enc_inputs)
print(dec_inputs)
print(dec_outputs)
```

```py
#自定义数据集函数
class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True) 
```

```py
d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度 
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8
```

```python
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()               # enc_inputs: [seq_len, d_model]
    def forward(self,enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1),:]
        return self.dropout(enc_inputs.cuda())
```

```python
def get_attn_pad_mask(seq_q,seq_k):
    batch_size, len_q = seq_q.size()# seq_q 用于升维，为了做attention，mask score矩阵用的
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size,len_q,len_k) # 扩展成多维度   [batch_size, len_q, len_k]
```

```python
def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask
```

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                             # Q: [batch_size, n_heads, len_q, d_k]
                                                                       # K: [batch_size, n_heads, len_k, d_k]
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                           # 如果是停用词P就等于 0 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        return context, attn
```

```pyhon
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):    # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn
```

```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        
    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)   # [batch_size, seq_len, d_model]  
```

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                                # enc_inputs: [batch_size, src_len, d_model]
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, d_model], 
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]                                                                   
        enc_outputs = self.pos_ffn(enc_outputs)                                       # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
```

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
```

```py
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask): # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                       # enc_outputs: [batch_size, src_len, d_model]
                                                                                       # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                       # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                 dec_inputs, dec_self_attn_mask)   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, 
                                                enc_outputs, dec_enc_attn_mask)    # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                    # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn
```

```py
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda() # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda() # [batch_size, tgt_len, tgt_len]

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，
        # 要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
```

```python
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder().cuda()
        self.Decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):                         # enc_inputs: [batch_size, src_len]  
                                                                       # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)         # enc_outputs: [batch_size, src_len, d_model], 
                                                                       # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                       # dec_outpus    : [batch_size, tgt_len, d_model], 
                                                                       # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], 
                                                                       # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                      # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```

```python
model = Transformer().cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)     #忽略 占位符 索引为0.
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
```

```python
for epoch in range(1000):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        
        loss = criterion(outputs,dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch+1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

# transformer训练mnist

```python
import numpy as np
import torch
import torch.nn as nn  #调用Linear、Conv2d等函数
from torch.autograd import Variable   #使变量可导
import torch.utils.data as date  #数据
import torchvision   #处理数据
#from torch_geometric.nn import GCNConv
#from torch_scatter import scatter_mean
device=torch.device('cuda:0')
```

```markdown
d_model –编码器/解码器输入大小（默认 512）。

nhead –多头注意力模型的头数（默认为8）。

num_encoder_layers –编码器中子编码器层的数量（默认为6）。

num_decoder_layers –解码器中子解码器层的数量（默认为6）。

dim_feedforward –前馈网络模型的中间层维度（默认= 2048）。

dropout –默认值= 0.1。

activation–编码器/解码器中间层的激活函数，relu或gelu（默认值= relu）。

custom_encoder –自定义编码器（默认=None）。

custom_decoder –自定义解码器（默认=None）。

layer_norm_eps - 层归一化组件中的eps值 (默认=1e-5).

batch_first: If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

norm_first: 如果“True”，编码器和解码器层将在其他注意和前馈操作之前执行LayerNorms，否则在其他注意和前馈操作之后执行LayerNorms。默认值：`False`（后面）。
```

```markdown
S是源序列长度，T是目标序列长度，N是批处理大小，E是特征编号

src: (S, N, E).

tgt: (T, N, E).

output:(T, N, E).
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
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.tsf=nn.Transformer(28,2,1,1,64,batch_first=True)
        self.fo=nn.Linear(10,28)
        self.f2=nn.Linear(28,10)
    def forward(self,x,y):
        #y=nn.functional.one_hot(y.long(),10)
        x=x.reshape(-1,28,28)
        #y=self.fo(y.float())
        #y=y[:,None,:]
        out=self.tsf(x,y)
        out=out.reshape(-1,28)
        out=self.f2(out)
        return out
    def init_hidden(self):
        pass
#net=Net()#.to(device)
#x=torch.zeros(32,1,28,28)
#y=torch.ones(32,1,28)
#a=net(x,y)
#a.shape
```

```python
def train(model,learn_rate,train_set,test_set,epoch):
    optimizer=torch.optim.Adam(model.parameters(),learn_rate)#Adam
    cost=nn.CrossEntropyLoss()#交叉熵
    for i in range(epoch):
        for x,y in train_set:
            yh=torch.ones(1,1,28)
            x=Variable(x.float(),requires_grad=True).to(device)
            yh=Variable(yh.float(),requires_grad=True).to(device)
            y=Variable(y.float()).to(device)
            optimizer.zero_grad()
            out=model(x,yh)
            loss=cost(out,y.long())   #求损失
            loss.backward()    #反向传播
            optimizer.step()    #更新
        print("epoch"+str(i+1)+"：")
    train_acc(model,train_set)
    test_acc(model,test_set)
    print("Finished training")

def train_acc(model,train_set):
    correct=0
    total=0
    for datas in train_set:
        x,y=datas
        yh=torch.ones(1,1,28)
        x=Variable(x).to(device)
        yh=Variable(yh.float()).to(device)
        y=y.to(device)
        a=torch.max(model(x,yh).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('train_acc:%.2f %%' % (100*correct/total))
def test_acc(model,test_set):
    correct=0
    total=0
    for datas in test_set:
        x,y=datas
        yh=torch.ones(1,1,28)
        x=Variable(x).to(device)
        yh=Variable(yh.float()).to(device)
        y=y.to(device)
        a=torch.max(model(x,yh).data,1)[1]
        total+=y.size(0)
        correct+=(a==y).sum()
    print('test_acc:%.2f %%' % (100*correct/total))
```

```python
net=Net().to(device)
batch_size=1
learn_rate=0.001
epoch=10
train_set,test_set=data_loader(batch_size)
train(net,learn_rate,train_set,test_set,epoch)
```

```python
epoch1：
epoch2：
epoch3：
epoch4：
epoch5：
epoch6：
epoch7：
epoch8：
epoch9：
epoch10：
train_acc:81.05 %
test_acc:80.88 %
Finished training
```

