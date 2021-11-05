import torch as t
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import nms

def bbox_encode(bbox,feature_map_size,img_size):
    hh,ww = feature_map_size
    h,w = img_size

    # print(bbox.size())
    bbox_h = bbox[:,3] - bbox[:,1]
    bbox_w = bbox[:,2] - bbox[:,0]

    bbox_cx = bbox_w/2 + bbox[:,0]
    bbox_cy = bbox_h/2 + bbox[:,1]

    bbox_encode = t.zeros((bbox.size()[0],5,4,4))

    iy = np.arange(0,h,h//hh)
    ix = np.arange(0,w,w//ww)

    for i in range(bbox.size()[0]):
        bbox_encode[i, 0, np.where(iy <= float(bbox_cy[i]))[0][-1], np.where(ix <= float(bbox_cx[i]))[0][-1]] = 1.
        bbox_encode[i,1:5,np.where(iy<=float(bbox_cy[i]))[0][-1],np.where(ix<=float(bbox_cx[i]))[0][-1]] = bbox[i]/h

    return bbox_encode

def bbox_decode(objects,score,loc,feature_map_size): # 只处理batch_size=1
    objects = objects.view(feature_map_size**2)

    score = t.argmax(score,dim=1)

    # print(loc)
    loc = loc.view(4,feature_map_size,feature_map_size)
    loc = loc.permute(1,2,0).contiguous()
    loc = loc.view(feature_map_size**2,feature_map_size)

    o_max = t.max(objects,dim=0)[1]
    loc = loc[o_max,:]


    # print(score)
    # print(loc)

    return score,loc








if __name__ == '__main__':
    from dataset import tiny_dataset
    dataset = tiny_dataset()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,batch_size=2,shuffle=False)
    for each in dataloader:
        bbox = each['bbox']
        break
    print(bbox.shape)
    gt = bbox_encode(bbox, (4, 4), (128, 128))
    print(bbox)
    print(gt)
    print(gt.shape)
    batch_size = bbox.size()[0]
    ww = 4
    hh = 4
    for i in range(batch_size):
        for ix in range(ww):
            for iy in range(hh):
                if gt[i, 0, iy, ix] == 1.:
                    print(gt[i,1:5,iy,ix])