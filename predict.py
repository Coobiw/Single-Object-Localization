from bbox_codec import bbox_decode
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import tiny_dataset
import argparse

def args_parser():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--load-model',help='the path pf .pth file of model',type=str,required=True,dest='load_model')

    return parser

if __name__ == '__main__':
    dataset = tiny_dataset()
    item = dataset[715]
    img = item['img']
    img = img.view(1,*img.size())
    args = args_parser().parse_args()
    print('loading ... ')
    net = t.load(args.load_model).cpu()
    print('predicting ... ')
    net.eval()
    with t.no_grad():
        objects,score,loc = net(img)
    score,loc = bbox_decode(objects,score,loc,4)
    print(score)

    img = img.view(3,128,128).numpy().transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = img * 255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    loc = loc.numpy()*128
    loc = loc.astype('uint8')

    classes = ['bird','car','dog','lizard','turtle']
    colors = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250)]
    classification = classes[score[0]]
    color = colors[score[0]]
    bbox_loc = loc

    cv2.rectangle(img,(bbox_loc[0],bbox_loc[1]),(bbox_loc[2],bbox_loc[3]),color=color)
    cv2.putText(img, classification, (bbox_loc[0],bbox_loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color = color)

    # cv2.imwrite('./result1.png',img)
    img = cv2.resize(img,(640,480))
    cv2.imshow('result',img)
    cv2.waitKey(0)
