import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import torch as t
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class tiny_dataset(Dataset):
    def __init__(self,root=os.path.join(os.getcwd(),'tiny_vid')):
        self.root = root
        data_dir = []
        label_dir = []
        for each in os.listdir(self.root):
            if '.txt' in each:
                label_dir.append(each)
            elif '.md' not in each:
                data_dir.append(each)
        self.imgs = read_img(root=self.root,data_dir=data_dir)
        self.labels = get_label() # （900，）的float64的Tensor
        self.bboxs = read_bbox(root=self.root,label_dir=label_dir) # （900，4）的float64的Tensor

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]
        )
        return {'img':img_transform(self.imgs[index]),
                'label':self.labels[index],'bbox':self.bboxs[index]}

def read_img(root,data_dir):
    img_path = []
    for each in data_dir:
        for img in os.listdir(os.path.join(root, each))[:180]:
            img_path.append(os.path.join(os.path.join(root, each), img))
    imgs = [Image.open(i) for i in img_path]
    return imgs

def get_label():
    l1 = np.zeros((180,))
    l2 = np.ones((180,))
    l3 = l2.copy() * 2
    l4 = l2.copy() * 3
    l5 = l2.copy() * 4

    labels = np.concatenate((l1, l2, l3, l4, l5))
    # print(labels)
    labels = t.from_numpy(labels)

    return labels

def read_bbox(root,label_dir):
    bbox_path = []
    for each in label_dir:
        bbox_path.append((os.path.join(root, each)))

    for each in bbox_path:
        if each == bbox_path[0]:
            bbox_data = pd.read_csv(each, sep=' ', header=None, index_col=None)
            bbox_data[0] -= 1
            bbox_data = bbox_data.iloc[:180]
            bbox_data = bbox_data.drop(0, axis=1)
            bbox_data = bbox_data.values
        else:
            data = pd.read_csv(each, sep=' ', header=None, index_col=None)
            data[0] -= 1
            data = data.iloc[:180]
            data = data.drop(0, axis=1)
            data = data.values
            bbox_data = np.concatenate((bbox_data, data), axis=0)
    bbox_data = t.from_numpy(bbox_data)

    return bbox_data

if __name__ == '__main__':
    dataset = tiny_dataset()
    item = dataset[899]
    img = item['img']
    bbox = item['bbox'].data.numpy()
    print(bbox.shape)
    img = img.numpy().transpose(1,2,0)
    img = img *[0.229,0.224,0.225] + [0.485,0.456,0.406]
    img = img*255
    img = img.astype('uint8')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    print(img.shape)
    # cv2.imshow('demo', img)
    # cv2.waitKey(0)
    img_bbox = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=[0, 0, 255])
    # img_bbox = cv2.rectangle(img,(bbox[0,0],bbox[0,1]),(bbox[0,2],bbox[0,3]),color=[0,0,255])

    cv2.imshow('demo',img_bbox)
    cv2.waitKey(0)