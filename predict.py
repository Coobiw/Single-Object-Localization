from bbox_codec import bbox_decode
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset import tiny_dataset
import argparse
import torchvision.transforms as transform
from PIL import Image

def args_parser():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--load-model',help='the path pf .pth file of model',type=str,required=True,dest='load_model')
    parser.add_argument('--img-path',help='the path of predicted image',type=str,default='None')
    return parser

def get_image_data(img_path):
    img = Image.open(img_path).convert('RGB').resize((128,128))
    transforms = transform.Compose(
        [
            transform.ToTensor(),
            transform.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ]
    )
    return transforms(img),np.array(img,dtype='uint8')

if __name__ == '__main__':
    args = args_parser().parse_args()

    # 这里是直接从数据集里选了一张做predict，事实上可以自己写get_image_data函数来获得（要返回Tensor）
    #img = get_img_data()
    if args.img_path == 'None':
        dataset = tiny_dataset()
        item = dataset[715]
        img = item['img']
        img = img.view(1,*img.size())
        img_np = img.view(3, 128, 128)
        img_np = img_np.numpy().transpose(1, 2, 0)
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_np = img_np * 255
        img_np = img_np.astype('uint8')
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # 网络加载和predict运算，得到分类的score和回归的坐标

    else:
        img , img_np  = get_image_data(args.img_path)
        img = img.view(1,*img.size())
        img_np = cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)


    print('loading model ... ')
    net = t.load(args.load_model).cpu()
    print('predicting ... ')
    net.eval()
    with t.no_grad():
        objects,score,loc = net(img)
    score,loc = bbox_decode(objects,score,loc,4)

    loc = loc.numpy()*128
    loc = loc.astype('uint8')

    classes = ['bird','car','dog','lizard','turtle']
    colors = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250)]
    classification = classes[score[0]]
    color = colors[score[0]]
    bbox_loc = loc

    cv2.rectangle(img_np,(bbox_loc[0],bbox_loc[1]),(bbox_loc[2],bbox_loc[3]),color=color)
    cv2.putText(img_np, classification, (bbox_loc[0],bbox_loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color = color)

    # cv2.imwrite('./result1.png',img)
    img_np = cv2.resize(img_np,(640,480))
    cv2.imshow('result',img_np)
    cv2.waitKey(0)
