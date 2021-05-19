import os
import datetime
from tqdm import tqdm
import torch
from Dataset.datasets import CelebA
from tensorboardX import SummaryWriter
import numpy as np
import random
from Experiments import ISSUE_16_EXP7
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from Model.PSAattenNet import PSANet
from Tools.utils import print_network

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.color import rgb2gray

####### EXPERIMENTS DEFINE #######################################
args = ISSUE_16_EXP7('test')   
# 
# To manage experiments and results. Please use GITHUB ISSUE BOARD :)
# (https://github.com/suzoosuagr/Demask_GAN/issues) 
# Example at
# https://github.com/suzoosuagr/Demask_GAN/issues/8 
# 
##################################################################

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if __name__ == "__main__":
    # Prepare TF writer
    if args.summary:
        if args.mode in args.summary_register:
            if not os.path.isdir(args.summary_dir):
                os.mkdir(args.summary_dir)
            summary_dir = os.path.join(args.summary_dir, args.model, args.name+ '_fold' + '/', datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            writer = SummaryWriter(summary_dir)
            args.save_config()
        else:
            writer = None
    else:
        writer = None

    # device
    if not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    # Prepare dataset
    if args.mask_type == "face_mask":
        with open("./Dataset/CelebA/face_mask/legi_test.txt", 'r') as f:
            lines = f.readlines()
            legi_test = [l.rstrip() for l in lines]

    else:
        raise ValueError("Mask_type [%s] not recognized. Please choose among ['face_mask', 'cnt_mask', 'irr_mask']  " % args.mask_type)



    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std = [0.5] * 3)
    ])
    mask_transforms = transforms.ToTensor()

    if args.mask_type == "face_mask":
        mask_dir = args.mask_dir
        img_dir = args.img_dir

    else:
        raise ValueError("Mask_type [%s] not recognized. Please choose among ['face_mask', 'cnt_mask', 'irr_mask']  " % args.mask_type)

    # dataset_train = CelebA(args.data_dir, args.img_dir, args.mask_dir, legi_train, augmentation=True, transform=img_transforms, mask_transform=mask_transforms)
    # dataset_eval = CelebA(args.data_dir, args.img_dir, args.mask_dir, legi_train, augmentation=False, transform=img_transforms, mask_transform=mask_transforms)
    dataset_test = CelebA("test", args.data_dir, img_dir, mask_dir, legi_test, args.sizes, args.mask_type, transform=img_transforms, mask_transform=mask_transforms)
    # dataset_test = CelebA(args.data_dir, args.img_dir, args.mask_dir, legi_test, augmentation=False, transform=img_transforms, mask_transform=mask_transforms)
    test_loader = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=12, pin_memory=True, drop_last=True)


    save_dir = os.path.join(args.visual_dir, args.mode, args.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # PREPARE MODEL ...
    model = PSANet(args.in_ch, args.out_ch, args, isTrain=False)
    model = model.to(device)

    print_network(model)
    start_epoch = 0
    total_steps = 0
    epoch = int(args.which_epoch)
    ssim_list = []
    psnr_list = []

    # Claim 
    print("WORKING UNDER {}".format(args.mode))

    # test
    for data in tqdm(test_loader):
        gt_img, img, mask = data

        total_steps += args.batch_size

        model.set_input(img, gt_img, mask, device)
            # model.optimize_parameters()
        model.test()

        img, mask, fake_B, real_B = model.get_current_visual()
            
        if total_steps % 1 == 0:
            mask = mask.expand(img.size())
            pic = ( torch.cat([img, fake_B, real_B], dim=0) + 1 ) / 2.0
            grid_pic = torchvision.utils.make_grid(pic, nrow=args.batch_size)
            torchvision.utils.save_image(grid_pic, os.path.join(save_dir, "Dddd_Epoch_{}_({}).png".format(epoch, total_steps)))

        for r, f in zip(real_B, fake_B):


            r = ((r + 1) / 2.0).squeeze().permute(1, 2, 0).cpu().numpy()
            f = ((f + 1) / 2.0).squeeze().permute(1, 2, 0).cpu().numpy()
            gray_rb = rgb2gray(r)
            gray_fb = rgb2gray(f)

            psnr_list.append(psnr(gray_rb, gray_fb, data_range=1))
            ssim_list.append(ssim(gray_rb, gray_fb, data_range=1, win_size=51, multichannel=True))


    print("mean ssim over test data is : {}".format(np.mean(ssim_list)))
    print("mean psnr over test data is : {}".format(np.mean(psnr_list)))
