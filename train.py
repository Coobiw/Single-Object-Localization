from dataset import tiny_dataset
from bbox_codec import bbox_encode
from resnet50_base import Localization_net2
from torch.utils.data import DataLoader,random_split
import torch as t
import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import argparse
from loss import Loss_for_localization
from evaluate import compute_three_acc
import os
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',help='learning rate',type=float,default=1e-2,dest='lr')
    parser.add_argument('--batch-size',help='batchsize',type=int,default=32,dest='batch_size')
    parser.add_argument('--lr-decay',help='the decay of lr',type=float,default=0.1,dest='lr_decay')
    parser.add_argument('--root',help='root directory of dataset',type=str,
                        default=r'E:\BS_learning\4_1\CV_basis\experiment\2\tiny_vid',dest='root')
    parser.add_argument('--weight-decay',help='weight decay of optimizer',type=float,
                        default=1e-5,dest='weight_decay')
    parser.add_argument('--epochs',help='set the num of epochs',type=int,default=100)
    parser.add_argument('--log-dir',help='tensorboard log dir',type=str,required=True)
    parser.add_argument('--save-file-name', help='the pth file name', type=str,required=True)
    parser.add_argument('--class-weight',help='the weight of classification of the loss',default=1,type=int)
    parser.add_argument('--regre-weight', help='the weight of regression of the loss', default=2,type=int)
    return parser

def weight_init(net):
    for name,child in net.named_children():
        if name == 'feature_extraction':
            continue

        if isinstance(child,nn.Conv2d):
            nn.init.kaiming_normal_(child.weight)
            if child.bias != None:
                nn.init.zeros_(child.bias)

        elif isinstance(child,nn.Linear):
            nn.init.kaiming_normal_(child.weight)
            if child.bias != None:
                nn.init.zeros_(child.bias)
    return net

def train():
    args = parser().parse_args()
    t.manual_seed(777)
    t.cuda.manual_seed(777)

    dataset = tiny_dataset(root=args.root)
    train_set,val_set = random_split(dataset=dataset,lengths=[150*5,30*5],
                                     generator=t.Generator().manual_seed(777))

    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True,num_workers=2)

    val_loader = DataLoader(dataset=val_set,batch_size=1,shuffle=False,num_workers=0)

    print('establish the net ...')
    net = Localization_net2(class_num=5).cuda()

    print('initialize the net')
    net = weight_init(net=net)
    high_lr_list = []
    low_lr_list = []
    for name,param in net.named_parameters():
        if 'feature_extraction' in name:
            low_lr_list.append(param)
        else:
            high_lr_list.append(param)

    optimizer = optim.SGD([{'params':low_lr_list,'lr':0.1*args.lr},{'params':high_lr_list}],
                          lr=args.lr,weight_decay=args.weight_decay,momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                             mode='min', patience=2,factor=args.lr_decay)


    writer = SummaryWriter(log_dir=args.log_dir,comment='curves_log')
    criterion = Loss_for_localization().cuda()

    for i in tqdm.tqdm(range(args.epochs)):
        t_loss = 0.
        tc_acc = 0.
        tr_acc = 0.
        t_acc = 0.

        v_loss = 0.
        vc_acc = 0.
        vr_acc = 0.
        v_acc = 0.

        print('\n%dth epoch'%(i+1))


        if i+1 == args.epochs//4:
            optimizer.param_groups[0]['lr'] *= args.lr_decay
            optimizer.param_groups[1]['lr'] *= args.lr_decay

        if i+1 == args.epochs//2:
            optimizer.param_groups[0]['lr'] *= args.lr_decay
            optimizer.param_groups[1]['lr'] *= args.lr_decay

        if i+1 == 3*args.epochs//4:
            optimizer.param_groups[0]['lr'] *= args.lr_decay
            optimizer.param_groups[1]['lr'] *= args.lr_decay

        for item in train_loader:
            tc_acc_num = 0
            tr_acc_num = 0
            t_acc_num = 0
            net.train()
            img = item['img'].cuda()
            label = item['label'].cuda()
            bbox = item['bbox'].cuda()
            objects, scores, locs = net(img)


            gt = bbox_encode(bbox=bbox,feature_map_size=(4,4),img_size=(128,128)).cuda()

            loss = criterion(objects,scores,locs,label,gt,args.regre_weight,0.5,args.class_weight)

            t_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            for j in range(img.size()[0]):
                a,b,c = compute_three_acc(objects=objects[j].view(1,*objects[j].size()),
                score=scores[j].view(1,*scores[j].size()), loc=locs[j].view(1,*locs[j].size()),
                label=label[j].view(1,*label[j].size()), bbox=bbox[j].view(1,*bbox[j].size()))
                tc_acc_num += a
                tr_acc_num += b
                t_acc_num += c

            tc_acc += tc_acc_num/float(img.size()[0])
            tr_acc += tr_acc_num / float(img.size()[0])
            t_acc += t_acc_num / float(img.size()[0])




        net.eval()
        with t.no_grad():
            for item2 in val_loader:
                img = item2['img'].cuda()
                label = item2['label'].cuda()
                bbox = item2['bbox'].cuda()
                objects, scores, locs = net(img)
                class_acc,regression_acc,acc = compute_three_acc(objects=objects,score=scores,
                                                        loc=locs,label=label,bbox=bbox)
                gt = bbox_encode(bbox=bbox, feature_map_size=(4, 4), img_size=(128, 128)).cuda()

                vc_acc += class_acc
                vr_acc += regression_acc
                v_acc += acc

                loss = criterion(objects, scores, locs,label, gt,args.regre_weight,0.5,args.class_weight)
                v_loss +=loss.item()

        v_loss /= len(val_loader)
        vc_acc /= len(val_loader)
        vr_acc /= len(val_loader)
        v_acc /= len(val_loader)

        # scheduler.step(v_loss)

        print('train_loss: %.5f  val_loss : %.5f' % (t_loss/len(train_loader),v_loss))

        writer.add_scalar('low_lr_curve', optimizer.param_groups[0]["lr"], i + 1)
        writer.add_scalar('high_lr_curve', optimizer.param_groups[1]["lr"], i + 1)
        writer.add_scalars('loss', {'Train':t_loss / len(train_loader)}, i+1)
        writer.add_scalars('loss', {'Val':v_loss}, i+1)
        writer.add_scalars('train_acc', {'class_acc': tc_acc/ len(train_loader)}, i + 1)
        writer.add_scalars('train_acc', {'regression_acc': tr_acc/ len(train_loader)}, i + 1)
        writer.add_scalars('train_acc', {'two_task_acc': t_acc/ len(train_loader)}, i + 1)
        writer.add_scalars('val_acc',{'class_acc':vc_acc},i+1)
        writer.add_scalars('val_acc', {'regression_acc': vr_acc}, i + 1)
        writer.add_scalars('val_acc', {'two_task_acc': v_acc}, i + 1)

        if optimizer.param_groups[0]['lr'] <= 1e-8:
            break
    t.save(net,os.path.join(args.log_dir,args.save_file_name + 'epoch%d.pth'%i))




if __name__ == '__main__':
    train()