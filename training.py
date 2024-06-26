#!/usr/bin/env python

"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""

# If this is the main scripts, we need gnd_net within our path as a module
# This is to make all imports compatible with both ros and ros free envrionments
if __name__ == '__main__':
    import sys
    sys.path.insert(1, "..") # Put the parent right behind the current directory (puts it before all ROS packages)

# import tracemalloc

# tracemalloc.start()


import argparse
import os
import shutil
import yaml
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import cv2

# from modules import gnd_est_Loss
from gnd_net.model import GroundEstimatorNet
from gnd_net.modules.loss_func import MaskedHuberLoss,SpatialSmoothLoss
from gnd_net.dataset_utils.dataset_provider import get_train_loader, get_valid_loader
from gnd_net.utils.point_cloud_ops import points_to_voxel
# # import ipdb as pdb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

use_cuda = torch.cuda.is_available()

# Init logging
logging.basicConfig()
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger_main = logging.getLogger().getChild('main')
logger_main.setLevel(logging.DEBUG)

fh = logging.FileHandler('training.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger_main.addHandler(fh)

if use_cuda:
    logger_main.info(f'setting gpu on gpu_id: {torch.cuda.get_device_name()}') #TODO: find the actual gpu id being used



#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


parser = argparse.ArgumentParser()
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', default='config/config_custom_local.yaml', type=str, metavar='PATH', help='path to config file (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-s', '--save_checkpoints', dest='save_checkpoints', action='store_true',help='evaluate model on validation set')
parser.add_argument('--start_epoch', default=0, type=int, help='epoch number to start from')
args = parser.parse_args()

if os.path.isfile(args.config):
    logger_main.info("using config file: %s", args.config)
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use

else:
    logger_main.warn("=> no config file found at '{}'".format(args.config))

#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


train_loader =  get_train_loader(cfg.data_dir, cfg.batch_size, skip = 6, num_input_features=cfg.input_features, max_memory=15e9, parent_logger=logger_main)
valid_loader =  get_valid_loader(cfg.data_dir, cfg.batch_size, skip = 3, num_input_features=cfg.input_features, max_memory=5e9, parent_logger=logger_main)

attempts = 10
attempt = 0

logger_main.info('Load model to GPU')
while (attempt <= attempts):
    try:
        model = GroundEstimatorNet(cfg).cuda()
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

        lossHuber = nn.SmoothL1Loss(reduction = "mean").cuda()
        lossSpatial = SpatialSmoothLoss().cuda()
        break
    except Exception as e:
        print(f'Failed to load GPU. Might be busy [{attempt}/{attempts}]: {e}')
        attempt += 1
        time.sleep(attempt * 2 + 2)

def train(epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()

    for batch_idx, (data, labels) in enumerate(train_loader):

        data_time.update(time.time() - start) # measure data loading time
        B = data.shape[0] # Batch size
        N = data.shape[1] # Num of points in PointCloud

        voxels = []; coors = []; num_points = []; mask = []
        # kernel = np.ones((3,3),np.uint8)

        data = data.numpy()
        for i in range(B):
            v, c, n = points_to_voxel(data[i], cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
            # m = np.zeros((100,100),np.uint8)
            # ind = c[:,1:]
            # m[tuple(ind.T)] = 1
            # m = cv2.dilate(m,kernel,iterations = 1)

            c = torch.from_numpy(c)
            c = F.pad(c, (1,0), 'constant', i)
            voxels.append(torch.from_numpy(v))
            coors.append(c)
            num_points.append(torch.from_numpy(n))
            # mask.append(torch.from_numpy(m))
# 
        voxels = torch.cat(voxels).float().cuda()
        coors = torch.cat(coors).float().cuda()
        num_points = torch.cat(num_points).float().cuda()
        labels = labels.float().cuda()
        # mask = torch.stack(mask).cuda()

        optimizer.zero_grad()

        output = model(voxels, coors, num_points)
        # pdb.set_trace()
        
        #l = [cfg.alpha, lossHuber(output, labels), cfg.beta, lossSpatial(output)]
        loss = cfg.alpha * lossHuber(output, labels) + cfg.beta * lossSpatial(output) # l[0]*l[1]+l[2]*l[3] #
        #logger_main.debug(f'Loss: {l[0]}*{l[1]}+{l[2]}*{l[3]}={loss}')
        # loss = lossHuber(output, labels)
        # loss = masked_huber_loss(output, labels, mask)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)

        optimizer.step() # optimiser step must be after clipping bcoz optimiser step updates the gradients.

        losses.update(loss.item(), B)

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()


        if batch_idx % args.print_freq == 0:
            logger_main.debug('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg



def validate():

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # if args.evaluate:
    #     model.train()
    with torch.no_grad():
        start = time.time()

        for batch_idx, (data, labels) in enumerate(valid_loader):

            data_time.update(time.time() - start) # measure data loading time
            B = data.shape[0] # Batch size
            N = data.shape[1] # Num of points in PointCloud

            voxels = []; coors = []; num_points = []; mask = []
            # kernel = np.ones((3,3),np.uint8)

            data = data.numpy()
            for i in range(B):
                v, c, n = points_to_voxel(data[i], cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
                # m = np.zeros((100,100),np.uint8)
                # ind = c[:,1:]
                # m[tuple(ind.T)] = 1
                # m = cv2.dilate(m,kernel,iterations = 1)

                c = torch.from_numpy(c)
                c = F.pad(c, (1,0), 'constant', i)
                voxels.append(torch.from_numpy(v))
                coors.append(c)
                num_points.append(torch.from_numpy(n))
                # mask.append(torch.from_numpy(m))

            voxels = torch.cat(voxels).float().cuda()
            coors = torch.cat(coors).float().cuda()
            num_points = torch.cat(num_points).float().cuda()
            labels = labels.float().cuda()
            # mask = torch.stack(mask).cuda()

            optimizer.zero_grad()

            output = model(voxels, coors, num_points)
            # pdb.set_trace()

            loss = cfg.alpha * lossHuber(output, labels) + cfg.beta * lossSpatial(output)
            # loss = lossHuber(output, labels)
            # loss = masked_huber_loss(output, labels, mask)

            losses.update(loss.item(), B)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()



            if batch_idx % args.print_freq == 0:
                logger_main.debug('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       batch_idx, len(valid_loader), batch_time=batch_time, loss=losses))

    return losses.avg



lowest_loss = 1

def main():
    # rospy.init_node('pcl2_pub_example', anonymous=True)
    global args, lowest_loss
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger_main.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger_main.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger_main.warn("=> no checkpoint found at '{}'".format(args.resume))



    if args.evaluate:
        validate()
        return

    logger_main.info("Start training")
    for epoch in range(args.start_epoch, cfg.epochs):
        #logger_main.info(f'Current Memory: {tracemalloc.get_traced_memory()}')
        # adjust_learning_rate(optimizer, epoch)
        loss_t = train(epoch)

        # evaluate on validation set
        loss_v = validate()

        scheduler.step()



        if (args.save_checkpoints):
            # remember best prec@1 and save checkpoint
            is_best = loss_v < lowest_loss
            lowest_loss = min(loss_v, lowest_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best)




'''
Save the model for later
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
