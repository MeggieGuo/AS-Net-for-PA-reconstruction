# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main
   Project Name:    GAN+aux_loss+5x3+ Pixel2pixel
   Author :         Hengrong LAN
   Date:            2019/3/1
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2019/3/15:
-------------------------------------------------
"""
import logging.config
from tqdm import tqdm

from model.AS_Net_model import AS_Net
# from model import PA_nonlocal_unet

from visualizer import Visualizer
from skimage.measure import compare_ssim, compare_psnr
from Dataload import multichanneldata
# import pytorch_msssim
from pytorch_msssim import MSSSIM
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import *
import numpy as np
import scipy.stats as st
import click
import argparse

import logging
import datetime


def beijing(a, b):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


logging.Formatter.converter = beijing

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = argparse.ArgumentParser(description='Hyper-parameters management')
# network options
parser.add_argument('--in_channels', type=int, default=14, help='channel of Network input')
parser.add_argument('--dataset_pathr', type=str, default='./Dataset/0531/20200526/', help='path of dataset')
parser.add_argument('--vis_env', type=str, default='AS-Net', help='visualization environment')
parser.add_argument('--save_path', type=str, default='checkpoint/NEW/', help='path of saved model')
parser.add_argument('--file_name', type=str, default='ours_multich_1.ckpt', help='file name of saved model')
parser.add_argument('--learning_rate', type=int, default=0.005, help='learning rate')  # 0.005
parser.add_argument('--batch_size', type=int, default=16, help='batch_size of training')
parser.add_argument('--test_batch', type=int, default=16, help='batch_size of testing')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--loadcp', type=bool, default=False, help='if load model')
parser.add_argument('--num_epochs', type=int, default=800, help='the number of epoches')
args = parser.parse_args()

logging.config.fileConfig("./logging.conf")

# create logger
log = logging.getLogger()

def main():
    with torch.cuda.device(1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AS_Net(in_channels=args.in_channels).to(device)

        batch_time = AverageMeter()
        train_ssim_meter = AverageMeter()
        train_psnr_meter = AverageMeter()
        test_ssim_meter = AverageMeter()
        test_psnr_meter = AverageMeter()

        vis = Visualizer(env=args.vis_env)

        train_dataset = multichanneldata.ReconDataset0526(args.dataset_pathr, train=True)
        test_dataset = multichanneldata.ReconDataset0526(args.dataset_pathr, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

        smooth_L1 = nn.SmoothL1Loss()
        msssim = MSSSIM(channel=1)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.loadcp:
            checkpoint = torch.load(args.save_path + 'latest_' + args.file_name)
            start_epoch = checkpoint['epoch']
            print('%s%d' % ('training from epoch:', start_epoch))
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            args.learning_rate = checkpoint['curr_lr']

        cudnn.benchmark = True
        total_step = len(train_loader)

        best_metric = {'test_epoch': 0, 'test_ssim': 0, 'test_psnr': 0}
        log.info('train image num: {}'.format(train_dataset.__len__()))
        log.info('val image num: {}'.format(test_dataset.__len__()))

        end = time.time()
        for epoch in range(args.start_epoch, args.num_epochs):
            for batch_idx, (rawdata, reimage, bfimg) in enumerate(tqdm(train_loader)):
                rawdata = rawdata.to(device)
                reimage = reimage.to(device)
                bfimg = bfimg.to(device)

                fake_img, bf_feature, side = model(rawdata, bfimg)
                loss_pe = smooth_L1(fake_img, reimage)
                bf_loss = smooth_L1(bf_feature, reimage)
                loss = 5 * loss_pe + bf_loss
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ssim = compare_ssim(np.array(reimage[0, 0, :, :].cpu().detach()),
                                    np.array(fake_img[0, 0, :, :].cpu().detach()))
                train_ssim_meter.update(ssim)
                psnr = compare_psnr(np.array(reimage[0, 0, :, :].cpu().detach()),
                                    np.array(fake_img[0, 0, :, :].cpu().detach()),
                                    data_range=1)
                train_psnr_meter.update(psnr)

                # visualization and evaluation
                if (batch_idx + 1) % 5 == 0:
                    reimage = reimage.detach()
                    bfimg = bfimg.detach()
                    bf_feature = bf_feature.detach()
                    side = side.detach()
                    fake_img = fake_img.detach()
                    vis.img(name='ground truth', img_=255 * reimage[0])
                    vis.img(name='DAS image', img_=255 * bfimg[0])
                    vis.img(name='textural map', img_=255 * bf_feature[0])
                    vis.img(name='side_output', img_=255 * side[0])
                    vis.img(name='output', img_=255 * fake_img[0])

                batch_time.update(time.time() - end)
                end = time.time()

            log.info(
                'Epoch [{}], Start [{}], Step [{}/{}], Loss: {:.4f}, Time [{batch_time.val:.3f}({batch_time.avg:.3f})]'
                    .format(epoch + 1, args.start_epoch, batch_idx + 1, total_step, loss.item(),
                            batch_time=batch_time))

            vis.plot_multi_win(
                dict(
                    bfloss=bf_loss.item(),
                    loss_mse=loss_pe.item(),
                    total_loss=loss.item(),
                ))

            vis.plot_multi_win(dict(train_ssim=train_ssim_meter.avg, train_psnr=train_psnr_meter.avg))
            log.info('tain_ssim: {}, train_psnr: {}'.format(train_ssim_meter.avg, train_psnr_meter.avg))

            # Validata
            if epoch % 5 == 0:
                with torch.no_grad():
                    for batch_idx, (rawdata, reimage, bfimg) in enumerate(tqdm(test_loader)):
                        rawdata = rawdata.to(device)
                        reimage = reimage.to(device)
                        bfimg = bfimg.to(device)
                        outputs, bf_feature, side_test = model(rawdata, bfimg)
                        test_ms_ssim = msssim(outputs, reimage)

                        ssim = compare_ssim(np.array(reimage.cpu().squeeze()), np.array(outputs.cpu().squeeze()))
                        test_ssim_meter.update(ssim)
                        psnr = compare_psnr(np.array(reimage.cpu().squeeze()), np.array(outputs.cpu().squeeze()),
                                            data_range=1)
                        test_psnr_meter.update(psnr)

                        if (batch_idx + 1) % 2 == 0:
                            reimage = reimage.detach()
                            bf_feature = bf_feature.detach()
                            outputs = outputs.detach()
                            side_test = side_test.detach()
                            bfimg = bfimg.detach()
                            vis.img(name='Test: ground truth', img_=255 * reimage[0])
                            vis.img(name='Test: DASimage', img_=255 * bfimg[0])
                            vis.img(name='Test: textural map', img_=255 * bf_feature[0])
                            vis.img(name='Test: output', img_=255 * outputs[0])
                            vis.img(name='Test: side_output', img_=255 * side_test[0])

                    vis.plot_multi_win(dict(
                        test_ssim=test_ssim_meter.avg,
                        test_psnr=test_psnr_meter.avg,
                        test_msssim=test_ms_ssim.item()
                    ))
                    log.info('test_ssim: {}, test_psnr: {}'.format(test_ssim_meter.avg, test_psnr_meter.avg))

            # Decay learning rate
            if (epoch + 1) % 50 == 0:
                args.learning_rate /= 5
                update_lr(optimizer, args.learning_rate)

            torch.save({'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer,
                        'curr_lr': args.learning_rate,
                        },
                       args.save_path + 'latest_' + args.file_name
                       )

            if best_metric['test_ssim'] < test_ssim_meter.avg:
                torch.save({'epoch': epoch,
                            'model': model,
                            'optimizer': optimizer,
                            'curr_lr': args.learning_rate,
                            },
                           args.save_path + 'best_' + args.file_name
                           )
                best_metric['test_epoch'] = epoch
                best_metric['test_ssim'] = test_ssim_meter.avg
                best_metric['test_psnr'] = test_psnr_meter.avg
            log.info('best_epoch: {}, best_ssim: {}, best_psnr: {}'.format(best_metric['test_epoch'],
                                                                           best_metric['test_ssim'],
                                                                           best_metric['test_psnr']))


def calc_confidence_interval(samples, confidence_value=0.95):
    if type(samples) is list:
        samples = np.asarray(samples)
    assert isinstance(samples, np.ndarray), 'args: samples {} should be np.array'.format(samples)
    stat_accu = st.t.interval(confidence_value, len(samples) - 1, loc=np.mean(samples), scale=st.sem(samples))
    center = (stat_accu[0] + stat_accu[1]) / 2
    deviation = (stat_accu[1] - stat_accu[0]) / 2
    return center, deviation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_top=10):
        self.reset()
        _array = np.zeros(shape=(num_top)) + 0.01
        self.top_list = _array.tolist()

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

    def top_update_calc(self, val):
        # update the lowest or NOT
        if val > self.top_list[0]:
            self.top_list[0] = val
            # [lowest, ..., highest]
            self.top_list.sort()
        # update mean, deviation
        mean, deviation = calc_confidence_interval(self.top_list)
        best = self.top_list[-1]
        return mean, deviation, best


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
