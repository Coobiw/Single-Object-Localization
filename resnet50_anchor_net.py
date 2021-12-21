import torch as t
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchsummary import summary
class Localization_anchor_net(nn.Module):
    def __init__(self,class_num=5,pretrained=True,anchor_num_per_cell=3): # without the background
        self.class_num = class_num
        self.anchor_num_per_cell = anchor_num_per_cell

        super(Localization_anchor_net,self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        self.feature_extraction = nn.Sequential()
        for name,each in resnet50.named_children():
            if (not isinstance(each, nn.AdaptiveAvgPool2d)) and (not isinstance(each, nn.Linear)):
                self.feature_extraction.add_module(name=name,module=each)

        self.sliding_window = nn.Conv2d(2048,2048,3,1,1,bias=False)
        self.out1 = nn.Conv2d(2048,(4+1)*self.anchor_num_per_cell, 1, 1, 0, bias=True)

        self.classi_conv = nn.Conv2d(2048, 512, 1, 1, 0, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.out2 = nn.Linear(512,self.class_num,bias=True)

        self.dropout0 = nn.Dropout2d(0.5)

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.65)


    def forward(self,x):
        self.features = self.feature_extraction(x) # 2048 4 4

        self.features = self.dropout0(self.features)

        self.sw = F.relu(self.sliding_window(self.features))

        self.sw = self.dropout1(self.sw)
        self.output1 = self.out1(self.sw)
        self.output1 = self.output1.view(self.output1.size()[0],self.anchor_num_per_cell,5,4,4)

        self.class_features = F.relu(self.classi_conv(self.features))
        self.class_features = self.dropout2(self.class_features)
        self.output2 = self.out2(self.gap(self.class_features).view(-1,512))


        # self.locs = t.sigmoid(self.output1[:,:,1:5,:,:])
        self.locs = self.output1[:, :, 1:5, :, :]

        self.object = t.sigmoid(self.output1[:,:,0,:,:])

        self.scores = self.output2

        return self.object,self.scores,self.locs







if __name__ == '__main__':
    import time

    input = t.zeros((1,3,128,128)).cpu()
    net = Localization_anchor_net(class_num=5,pretrained=True,anchor_num_per_cell=3).cpu()
    start = time.time()
    object,score,loc = net(input)
    end = time.time()
    print(object.size())
    print(score.size())
    print(loc.size())
    print(end-start)
    
    print(summary(net,input_size=(3,128,128),device='cpu'))