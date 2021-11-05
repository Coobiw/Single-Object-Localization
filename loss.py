import torch.nn.functional as F
import torch.nn as nn
import torch as t

class Loss_for_localization(nn.Module):
    def __init__(self):
        super(Loss_for_localization,self).__init__()

    def forward(self,objects,scores,locs,label,gt,obj_coor,no_obj_confi,img_class_weight):
        batch_size,hh,ww = objects.size()
        no_obj_confi_loss = 0.
        obj_confi_loss = 0.
        img_class_loss = 0.
        obj_coor_loss = 0.

        for i in range(batch_size):
            for ix in range(ww):
                for iy in range(hh):
                    if gt[i,0,iy,ix] == 0.: #no_obj
                        # no_obj_confi_loss += objects[i,iy,ix]**2
                        no_obj_confi_loss += F.binary_cross_entropy(objects[i,iy,ix],gt[i,0,iy,ix])

                    elif gt[i,0,iy,ix] == 1.:
                        # obj_confi_loss += (objects[i,iy,ix]-bbox_iou(locs[i,0:4,ix,iy],gt[i,2:6,ix,iy]))**2
                        obj_confi_loss += F.binary_cross_entropy(objects[i,iy,ix],gt[i,0,iy,ix])

                        obj_coor_loss += ((locs[i,0:4,iy,ix]-gt[i,1:5,iy,ix])**2).sum(axis=0)

        img_class_loss = F.cross_entropy(scores,label.long())
        return img_class_weight*img_class_loss + \
               ((no_obj_confi*no_obj_confi_loss + obj_confi_loss + obj_coor * obj_coor_loss)/batch_size)