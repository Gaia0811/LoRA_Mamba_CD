import os
import time
import torch.autograd
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import OrderedDict

working_path = os.path.abspath('.')

from utils.loss import LatentSimilarity
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter

###################### Data and Model ########################
from models.Radio_Mamba import RadioMamba_v2 as Net
NET_NAME = 'Radio_Mamba'

from datasets import S2Look as RS
DATA_NAME = 'S2Look'
#from datasets import Levir_CD as RS
#DATA_NAME = 'LevirCD'
#from datasets import CLCD as RS
#DATA_NAME = 'CLCD'
#from datasets import WHU_CD_list as RS
#DATA_NAME = 'WHU_CD_0.05'
###################### Data and Model ########################


########################## Parameters ########################
args = {
    'train_batch_size': 8,
    'val_batch_size': 8,
    'lr': 0.1,
    'epochs': 50,
    'gpu': True,
    'dev_id': 0,
    'multi_gpu': None,  #"0,1,2,3",
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 10,
    'predict_step': 5,
    'crop_size': 512,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'effSAM_LoRA_highRes_e99_OA99.13_F90.06_IoU86.86.pth')}
########################## Parameters ########################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
writer = SummaryWriter(args['log_dir'])

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
        
def defreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def load_model(net):
    state_dict = torch.load(args['load_path'], map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
    net.load_state_dict(new_state_dict, strict=False) #
    return net

def main():
    net = Net()
    if args['multi_gpu']:
        net = torch.nn.DataParallel(net, [int(id) for id in args['multi_gpu'].split(',')])
    #load_model(net)
    net.to(device=torch.device('cuda', int(args['dev_id'])))
    #freeze_model(net)
    
    # test run
    dsize = (1, 3, 512, 512)
    x1 = torch.randn(dsize).cuda()
    x2 = torch.randn(dsize).cuda()
    net(x1, x2)
    torch.cuda.empty_cache()

    train_set = RS.RS('train', random_crop=True, crop_nums=6, crop_size=args['crop_size'], random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.RS('val', sliding_crop=True, crop_size=args['crop_size'], random_flip=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    params = list(filter(lambda p: p.requires_grad, net.parameters()))
             #+ list(filter(lambda p: p.requires_grad, net.sam.parameters()))
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, params), args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    train(train_loader, net, optimizer, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, optimizer, val_loader):
    bestF = 0.2
    bestacc = 0.0
    bestIoU = 0.0
    bestloss = 1.0
    bestaccT = 0.0

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    #criterion_sem = LatentSimilarity(T=3.0).to(torch.device('cuda', int(args['dev_id'])))
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, imgs_B, labels = data
            if args['gpu']:
                imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
                imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
                labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = net(imgs_A, imgs_B)

            assert outputs.shape[1] == 1
            loss_bn = F.binary_cross_entropy_with_logits(outputs, labels)
            #loss_t = criterion_sem(outA, outB, labels)
            loss = loss_bn #+ loss_t
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()
            preds = outputs.numpy()
            preds = F.sigmoid(outputs).numpy()
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, IoU = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))
                writer.add_scalar('train_loss', train_loss.val, running_iter)
                loss_rec = train_loss.val
                writer.add_scalar('train_accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        val_F, val_acc, val_IoU, val_loss = validate(val_loader, net, curr_epoch)
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME + '_e%d_OA%.2f_F%.2f_IoU%.2f.pth' % (
            curr_epoch, val_acc * 100, val_F * 100, val_IoU * 100)))
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        print('[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1 score: %.2f IoU %.2f' \
              % (curr_epoch, args['epochs'], time.time() - begin_time, bestaccT * 100, bestacc * 100, bestF * 100,
                 bestIoU * 100))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data

        if args['gpu']:
            imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
            imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
            labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            outputs = net(imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        preds = outputs.numpy()
        preds = F.sigmoid(outputs).numpy()
        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_color = RS.Index2Color(preds[0].squeeze())
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '.png'), pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f' % (
    curr_time, val_loss.average(), Acc_meter.average() * 100, F1_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', Acc_meter.average(), curr_epoch)

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg


def adjust_lr(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 3.0)
    running_lr = args['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()