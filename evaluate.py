from bbox_codec import bbox_decode
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import tiny_dataset
from torch.utils.data import random_split, DataLoader
import argparse

def args_parser():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--load-model',help='the path pf .pth file of model',type=str,required=True,dest='load_model')

    return parser

def bbox_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0

def compute_three_acc(objects, score, loc,label,bbox):
    regression_acc = 0
    class_acc = 0
    acc = 0
    score, loc = bbox_decode(objects, score, loc, 4)

    if score[0] == label[0]:
        class_acc += 1

    bbox_loc = loc
    iou = bbox_iou(t.round(128 * bbox_loc), bbox[0])
    # print(iou)
    if iou >= 0.5:
        regression_acc += 1

    if score[0] == label[0]:
        bbox_loc = loc
        iou = bbox_iou(t.round(128 * bbox_loc), bbox[0])
        # print(iou)
        if iou >= 0.5:
            acc += 1

    return class_acc , regression_acc , acc

if __name__ == '__main__':
    t.manual_seed(777)
    t.cuda.manual_seed(777)
    dataset = tiny_dataset()
    train_set, val_set = random_split(dataset=dataset, lengths=[150 * 5, 30 * 5],
                                      generator=t.Generator().manual_seed(777))
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    args = args_parser().parse_args()
    print('loading ... ')
    net = t.load(args.load_model).cpu()
    print('evaluating ... ')
    net.eval()
    with t.no_grad():
        total = 0
        regression_acc = 0
        class_acc = 0
        cm = 0
        acc = 0

        acc_index = []
        for i,item in enumerate(val_loader):
            img = item['img']
            label = item['label']
            bbox = item['bbox']
            objects, score, loc = net(img)
            score, loc = bbox_decode(objects, score, loc, 4)
            cm+=1
            if score[0] == label[0]:
                class_acc+=1

            bbox_loc = loc
            iou = bbox_iou(t.round(128*bbox_loc),bbox[0])
            # print(iou)
            if iou >= 0.5:
                regression_acc +=1

            if score[0] == label[0]:
                bbox_loc = loc
                iou = bbox_iou(t.round(128*bbox_loc),bbox[0])
                # print(iou)
                if iou >= 0.5:
                    acc +=1
                    acc_index.append(cm-1)

        # print(total)
        print(acc)
        # print(acc/total)
        # print(class_acc)
        # print(class_acc/len(val_loader))

