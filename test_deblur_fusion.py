"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
import skimage.measure
import scipy.io as sio
from networks.MIRNet_model import MIRNet,MIRNet_fusion
from dataloaders.data_rgb import get_test_data,get_test_images,get_test_images_fusion
import utils
from skimage import img_as_ubyte
import  skimage.measure

parser = argparse.ArgumentParser(description='RGB deblur evaluation on GP dataset')
parser.add_argument('--input_dir', default='./datasets/dnd/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/deblur/GP/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='/home2/zengwh/deblur/MIRNet/checkpoints/Deblur/models/MIRNetfusion32_b8_c64/model_latest.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='save denoised images in result directory')
parser.add_argument('--save_model_name', default='fusion', type=str, help='save_model_name')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


test_dataset = get_test_images_fusion('test')
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)



model_restoration = MIRNet_fusion(in_channels=3, out_channels=3, n_feat=32, kernel_size=3, stride=2, n_RRG=3, n_MSRB=2, height=3, width=2, bias=False)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()


with torch.no_grad():
    test_psnr = 0.
    test_ssim = 0.
    total_num = 0.
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_clean = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        rgb_dark = data_test[2].cuda()

        filenames = data_test[3]
        rgb_restored = model_restoration(rgb_noisy,rgb_dark)
        rgb_restored = torch.clamp(rgb_restored,0,1)
     
        #rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        #rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        gt = rgb_clean.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output = torch.clamp(rgb_restored, 0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)

        save_pic_name = os.path.join('result', args.save_model_name, filenames[0].split('/')[-2])
        os.makedirs(save_pic_name, exist_ok=True)
        output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(save_pic_name, filenames[0].split('/')[-1]), np.uint8(output * 255))
        total_num += 1
        test_ssim += (skimage.measure.compare_ssim(np.uint8(output[:, :, 0] * 255),
                                                                np.uint8(gt[:, :, 0] * 255), 255)
                                   + skimage.measure.compare_ssim(np.uint8(output[:, :, 1] * 255),
                                                                  np.uint8(gt[:, :, 1] * 255), 255)
                                   + skimage.measure.compare_ssim(np.uint8(output[:, :, 2] * 255),
                                                                  np.uint8(gt[:, :, 2] * 255), 255)) / 3
        test_psnr+= skimage.measure.compare_psnr(np.uint8(output * 255), np.uint8(gt * 255), 255)

    print('average psnr: ',test_psnr/total_num,'average ssim: ',test_ssim/total_num)



