import torch as t
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
class Localization_net2(nn.Module):
    def __init__(self,class_num=5): #加上背景
        self.class_num = class_num
        super(Localization_net2,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.feature_extraction = nn.Sequential()
        for name,each in resnet50.named_children():
            if (not isinstance(each, nn.AdaptiveAvgPool2d)) and (not isinstance(each, nn.Linear)):
                self.feature_extraction.add_module(name=name,module=each)

        self.sliding_window = nn.Conv2d(2048,2048,3,1,1,bias=False)
        self.out1 = nn.Conv2d(2048,5, 1, 1, 0, bias=True)

        self.classi_conv = nn.Conv2d(2048, 512, 1, 1, 0, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.out2 = nn.Linear(512,self.class_num,bias=True)

        # self.dropout01 = nn.Dropout2d(0.5)
        self.dropout02 = nn.Dropout2d(0.65)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.65)


    def forward(self,x):
        self.features = self.feature_extraction(x) # 2048 4 4
        # self.features01 = self.dropout01(self.features)
        self.features02 = self.dropout02(self.features)

        self.sw = F.relu(self.sliding_window(self.features))
        # self.sw = F.relu(self.sliding_window(self.features01))
        self.sw = self.dropout1(self.sw)
        self.output1 = self.out1(self.sw)

        self.class_features = F.relu(self.classi_conv(self.features02))
        self.class_features = self.dropout2(self.class_features)
        self.output2 = self.out2(self.gap(self.class_features).view(-1,512))

        self.locs = t.sigmoid(self.output1[:,1:5,:,:])

        self.object = t.sigmoid(self.output1[:,0,:,:])

        self.scores = self.output2

        return self.object,self.scores,self.locs







if __name__ == '__main__':
    vgg16 = models.vgg16(pretrained=True) # 5次下采样
    print(vgg16)
    # print(len(list(vgg16.features)))
    print(list(vgg16.features)[:30])
    # net = nn.Sequential()
    # for name,each in resnet18.named_children():
    #     if (not isinstance(each,nn.AdaptiveAvgPool2d)) and (not isinstance(each,nn.Linear)):
    #         net.add_module(name=name,module=each)
    # print(net)
    # input = t.zeros((1, 3, 128, 128)).cuda()
    # net = net.cuda()
    # print(net.maxpool(net.relu(net.bn1(net.conv1(input)))).size())

    # net = Localization_net5(class_num=5).cpu()
    # for name,child in net.named_children():
    #     if name == 'feature_extraction':
    #         print(child.conv1.bias)
        # print(type(child))



    # import time
    #
    # input = t.zeros((1,3,128,128)).cpu()
    # net = Localization_net2(class_num=5).cpu()
    # start = time.time()
    # object,score,loc = net(input)
    # end = time.time()
    # print(object.size())
    # print(score.size())
    # print(loc.size())
    # print(end-start)
    #
    # print(summary(net,input_size=(3,128,128),device='cpu'))
    # for name, param in net.named_parameters():
    #     print('name:',name)
    #     # print('param:',param)
    #
    # high_lr_list = []
    # low_lr_list = []
    # for name, param in net.named_parameters():
    #     if 'feature_extraction' in name:
    #         high_lr_list.append(param)
    #     else:
    #         low_lr_list.append(param)
    #
    # print(high_lr_list)
    # print(low_lr_list)