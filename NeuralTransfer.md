[toc]

# 出处

论文：《A Neural Algorithm of Artistic Style》[paper ](https://arxiv.org/pdf/1508.06576.pdf) [code](https://blog.51cto.com/u_14328065/4795718)

# 代码

```python
# load_img模块
import PIL.Image as Image
import torch
import torchvision.transforms as transforms

img_size = 512 if torch.cuda.is_available() else 128#根据设备选择改变后项数大小
def load_img(img_path):#图像读入
    img = Image.open(img_path).convert('RGB')#将图像读入并转换成RGB形式
    img = img.resize((img_size, img_size))#调整读入图像像素大小
    img = transforms.ToTensor()(img)#将图像转化为tensor
    img = img.unsqueeze(0)#在0维上增加一个维度
    return img

def show_img(img):#图像输出
    img = img.squeeze(0)#将多余的0维通道删去
    img = transforms.ToPILImage()(img)#将tensor转化为图像
    img.show()
```

```python
import torch.nn as nn
import torch

class Content_Loss(nn.Module):#内容损失
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()#继承父类的初始化
        self.weight = weight
        self.target = target.detach() * self.weight
        # 必须要用detach来分离出target，这时候target不再是一个Variable，这是为了动态计算梯度，否则forward会出错，不能向前传播
        self.criterion = nn.MSELoss()#利用均方误差计算损失

    def forward(self, input):#向前计算损失
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_graph=True):#反向求导
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class Gram(nn.Module):#定义Gram矩阵
    '''
    格拉姆矩阵可以看做feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵），在feature map中，
    每个数字都来自于一个特定滤波器在特定位置的卷积，因此每个数字代表一个特征的强度，
    而Gram计算的实际上是两两特征之间的相关性，哪两个特征是同时出现的，哪两个是此消彼长的等等，
    同时，Gram的对角线元素，还体现了每个特征在图像中出现的量，因此，Gram有助于把握整个图像的大体风格。
    有了表示风格的Gram Matrix，要度量两个图像风格的差异，只需比较他们Gram Matrix的差异即可。
    
    总之， 格拉姆矩阵用于度量各个维度自己的特性以及各个维度之间的关系。内积之后得到的多尺度矩阵中，
    对角线元素提供了不同特征图各自的信息，其余元素提供了不同特征图之间的相关信息。
    这样一个矩阵，既能体现出有哪些特征，又能体现出不同特征间的紧密程度。

    '''
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):#向前计算Gram矩阵
        a, b, c, d = input.size()#a为批量大小，b为feature map的数量，c*d为feature map的大小
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram


class Style_Loss(nn.Module):#风格损失
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        G = self.gram(input) * self.weight
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
```

```python
import torch.nn as nn
import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#选择运行设备，如果你的电脑有gpu就在gpu上运行，否则在cpu上运行
vgg = models.vgg19(pretrained=True).features.to(device)#这里我们使用预训练好的vgg19模型
#vgg.load_state_dict(torch.load('../task'))
'''所需的深度层来计算风格/内容损失:'''
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_loss(style_img,    #(1,3,512,512)
                             content_img,   #(1,3,512,512)
                             cnn=vgg,
                             style_weight=1000,
                             content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):
    content_loss_list = [] #内容损失
    style_loss_list = [] #风格损失
    model = nn.Sequential() #创建一个model，按顺序放入layer
    model = model.to(device)
    gram = Gram().to(device)

    '''把vgg19中的layer、content_loss以及style_loss按顺序加入到model中：'''
    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)
            if name in content_layers_default:
                target = model(content_img)
                content_loss = Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)
            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                style_loss = Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)
            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)
        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

    return model, style_loss_list, content_loss_list
```

```python
import torch.nn as nn
import torch.optim as optim

def get_input_param_optimier(input_img):
    """input_img is a Variable"""
    input_param = nn.Parameter(input_img.data)#获取参数
    optimizer = optim.LBFGS([input_param])#用LBFGS优化参数
    return input_param, optimizer

def run_style_transfer(content_img, style_img, input_img, num_epoches=300):
    print('Building the style transfer model..')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(
        style_img, content_img)
    input_param, optimizer = get_input_param_optimier(input_img)
    print('Opimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:#每隔50次输出一次loss
        def closure():
            input_param.data.clamp_(0, 1)#修正输入图像的值 返回值在该范围内小于0就返回0，大于1就返回1，在之间就返回它自己
            model(input_param)
            style_score = 0
            content_score = 0
            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()    
            for cl in content_loss_list:
                content_score += cl.backward()
            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(style_score.item(), content_score.item()))
                print()
            return style_score + content_score
        optimizer.step(closure)
    input_param.data.clamp_(0, 1)#再次修正
    return input_param.data
```

```python
from torch.autograd import Variable
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_img = load_img('./picture/style9.png')#风格图片地址   (1,3,512,512)
style_img = Variable(style_img).to(device)
content_img = load_img('./picture/content4.png')#内容图片地址  (1,3,512,512)
content_img = Variable(content_img).to(device)
input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)#进行200次训练
save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
save_pic.save('./picture/result11.png')#选择你要保存的地址
save_pic.show()
```

# 用法

注意save路径，在相应路径下添加一张content图片和style文件即可（文件像素越大效果越好）。