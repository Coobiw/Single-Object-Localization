import numpy as np
import torch as t

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

def kms_result_anchor(device):
    return t.tensor([[101.30232 ,  65.9907  ],
       [ 84.436264, 106.35694 ],
       [ 48.126507,  66.313255]],dtype = t.float32,device=device)

def xywh2xxyy(bbox):
    cx,cy,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
    return t.tensor([cx-w/2,cy-h/2,cx+w/2,cy+h/2],dtype=bbox.dtype,device=bbox.device)

def anchor_generate(kms_anchor,hh = 4,ww = 4,h = 128,w = 128):
    dtype = kms_anchor.dtype
    device = kms_anchor.device
    assert h == w , 'input image is not a square'

    # 归一化anchor的长宽
    kms_anchor = kms_anchor/h

    shifts_x = ((t.arange(0, ww)+0.5) / ww).to(dtype=dtype,device=device)
    shifts_y = ((t.arange(0, hh) + 0.5) / hh).to(dtype=dtype,device=device)
    shifts_x,shifts_y = t.meshgrid(shifts_x,shifts_y)
    shifts_x = shifts_x.reshape(-1,1)
    shifts_y = shifts_y.reshape(-1,1)
    anchor_list = []

    for i in range(kms_anchor.shape[0]):
        ah,aw = kms_anchor[i][0],kms_anchor[i][1]
        anchor = t.cat((shifts_x,shifts_y,t.ones(shifts_x.shape,dtype=dtype,device=device)*aw,
                        t.ones(shifts_x.shape,dtype=dtype,device=device)*ah),dim=1)
        anchor_list.append(anchor)

    # print(shifts_x,shifts_y)

    return t.cat(anchor_list,dim=0).view(3,16,4).permute(0,2,1).contiguous().view(3,4,4,4)

def anchor_encode(anchors,bbox,anchor_num_per_grid,feature_map_size,img_size,dtype,device): # bbox: (B,4)
    hh,ww = feature_map_size
    h,w = img_size
    assert h == w, 'input image is not a square'

    bbox_h = bbox[:, 3] - bbox[:, 1]
    bbox_w = bbox[:, 2] - bbox[:, 0]

    bbox_cx = bbox_w / 2 + bbox[:, 0]
    bbox_cy = bbox_h / 2 + bbox[:, 1]

    bbox_xywh = t.cat((bbox_cx.reshape(-1,1),bbox_cy.reshape(-1,1),
                       bbox_w.reshape(-1,1),bbox_h.reshape(-1,1)),dim=1)
    # print(bbox_xywh)

    gt = t.zeros((bbox.size()[0],anchor_num_per_grid,5,hh,ww),dtype=dtype,device=device)
    iy = np.arange(0, h, h // hh)
    ix = np.arange(0, w, w // ww)

    for i in range(bbox.size()[0]):
        iy = np.where(iy <= float(bbox_cy[i]))[0][-1]
        ix = np.where(ix <= float(bbox_cx[i]))[0][-1]
        best = 0.
        best_indice = -1
        for ia in range(anchor_num_per_grid):
            iou = bbox_iou(xywh2xxyy(anchors[ia,:,iy,ix]),xywh2xxyy(bbox_xywh[i]/h))
            if iou > best:
                best = iou
                best_indice = ia

        gt[i,best_indice, 0, iy,ix] = 1.
        gt[i,best_indice,1:5, iy,ix] = xywh2offset(anchors[best_indice,:,iy,ix],bbox_xywh[i]/h)

    return gt

def xywh2offset(anchor,bbox):
    acx,acy,aw,ah = anchor[0],anchor[1],anchor[2],anchor[3]
    gcx,gcy,gw,gh = bbox[0],bbox[1],bbox[2],bbox[3]
    offset0 = (gcx - acx) / acx
    offset1 = (gcy - acy) / acy
    offset2 = t.log(gw/aw)
    offset3 = t.log(gh/ah)

    return t.tensor([offset0,offset1,offset2,offset3])

def offset2xywh(anchor,offset):
    acx, acy, aw, ah = anchor[0], anchor[1], anchor[2], anchor[3]
    cx = offset[0]*acx + acx
    cy = offset[1]*acy + acy
    w = t.exp(offset[2]) * aw
    h = t.exp(offset[3]) * ah

    x1 = cx - w/2
    x2 = cx + w/2
    y1 = cy - h/2
    y2 = cy + h/2

    return t.tensor([x1,y1,x2,y2])

def anchor_decode(objects,scores,offsets,anchors):# offsets:[anchor_num_per_grid,4,hh,ww]
    # objects:[anchor_num_per_grid,hh,ww],score:[class_num]
    anchor_num_per_gird,_,hh,ww = offsets.shape

    objects = objects.view(anchor_num_per_gird*hh*ww)
    score = t.argmax(scores,dim=0)

    predicts = t.zeros(offsets.shape,dtype=offsets.dtype,device=offsets.device)

    for ia in range(anchor_num_per_gird):
        for ih in range(hh):
            for iw in range(ww):
                predicts[ia,:,ih,iw] = offset2xywh(anchors[ia,:,ih,iw],offsets[ia,:,ih,iw])

    predicts = predicts.permute(0, 2, 3, 1).contiguous().view(anchor_num_per_gird * hh * ww, 4)
    o_max = t.max(objects,dim=0)[1]

    loc = predicts[o_max,:]

    return loc,score

if __name__ == '__main__':
    anchors = anchor_generate(kms_anchor=kms_result_anchor(t.device('cuda')))
    print(anchors)
    print(anchors.shape)

    from dataset import tiny_dataset

    dataset = tiny_dataset()
    # print(len(dataset))
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for each in dataloader:
        bbox = each['bbox'].to('cuda')
        break

    gt = anchor_encode(anchors,bbox,3, (4, 4), (128, 128),t.float32,t.device('cuda'))
    print(gt[:,:,0])
    print(gt.shape)
    batch_size = bbox.size()[0]
    ww = 4
    hh = 4
    # for i in range(batch_size):
    #     for ix in range(ww):
    #         for iy in range(hh):
    #             if gt[i, 0, iy, ix] == 1.:
    #                 print(gt[i, 1:5, iy, ix])