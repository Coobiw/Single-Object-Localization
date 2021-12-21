import torch.nn.functional as F
import torch.nn as nn
import torch as t


class Loss_for_localization(nn.Module):
    def __init__(self,obj_coor,obj_confi,no_obj_confi,img_class_weight):
        super(Loss_for_localization,self).__init__()
        self.obj_coor = obj_coor
        self.no_obj_confi = no_obj_confi
        self.img_class_weight = img_class_weight
        self.obj_confi = obj_confi

    def forward(self,objects,scores,locs,label,gt):
        batch_size,anchor_num_per_cell,hh,ww = objects.size()
        no_obj_confi_loss = 0.
        obj_confi_loss = 0.
        obj_coor_loss = 0.

        for i in range(batch_size):
            for ia in range(anchor_num_per_cell):
                for iy in range(hh):
                    for ix in range(ww):
                        if gt[i,ia,0,iy,ix] == 0.: # no_obj anchor
                            no_obj_confi_loss += F.binary_cross_entropy(objects[i,ia,iy,ix],gt[i,ia,0,iy,ix])

                        elif gt[i,ia,0,iy,ix] == 1.:
                            obj_confi_loss += F.binary_cross_entropy(objects[i,ia,iy,ix],gt[i,ia,0,iy,ix])


                            obj_coor_loss += ((locs[i,ia,0:4,iy,ix]-gt[i,ia,1:5,iy,ix])**2).sum(axis=0)

        img_class_loss = F.cross_entropy(scores,label.long())
        return self.img_class_weight *img_class_loss + \
               ((self.no_obj_confi * no_obj_confi_loss + self.obj_confi * obj_confi_loss +
                 self.obj_coor * obj_coor_loss)/batch_size)