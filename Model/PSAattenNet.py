import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.networks import get_scheduler, Vgg16, PFDiscriminator, NLayerDiscriminator, Discriminator, MuliAttScheme
from .loss import GANLoss, ConsisLoss, StyleLoss, PerceptualLoss
import numpy as np


class PSANet(nn.Module):
    def __init__(self, in_ch, out_ch, args, isTrain=True):
        super(PSANet, self).__init__()
        self.isTrain = isTrain
        self.args = args

        # self.gener = HighResolutionNet(in_ch, out_ch)         # generator
        # self.gener = Unet_256_full(in_ch, args.base_dim, out_ch)
        self.gener = MuliAttScheme(args.in_ch+1, args.out_ch)
        self.vgg = Vgg16()                   # for NetF
        self.save_dir = os.path.join(args.ckpt_dir, args.name)

        # descriminators
        if self.isTrain:
            self.netD = NLayerDiscriminator(in_ch, args.base_dim, n_layers=3)
            self.netF = PFDiscriminator()
            # self.netD = Discriminator(in_channels=3, use_sigmoid=True)

        # # Initialization
        # if self.isTrain and not args.resume:
        #     self.apply(self.init_weights)\

        # Testing
        if not self.isTrain or args.resume:
            print("loading ckpt ... ")
            self.load_network(self.gener, 'PENnet', args.which_epoch)
            # resume
            if self.isTrain:
                self.load_network(self.netD, 'D', args.which_epoch)
                self.load_network(self.netF, 'F', args.which_epoch)

        if self.isTrain:
            self.old_lr = args.lr
            # Losses
            self.PerceptualLoss = PerceptualLoss()
            self.StyleLoss = StyleLoss()
            self.criterionDiv = nn.KLDivLoss()              # attention supervised 
            self.criterionMSE = nn.MSELoss()
            self.criterionGAN = GANLoss(target_real_label=0.9, target_fake_label=0.1)
            self.criterionL1 = nn.L1Loss()
            # self.adversarial_loss = set_device(AdversarialLoss(type=self.config['losses']['gan_type']))
            # optimizers
            self.schedulers = []
            self.optimizers = []
            # Coarse optimizer for v1 . .  
            self.optimizer_G = torch.optim.Adam(self.gener.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)

            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, args))

    def set_input(self, img, gt, mask, device):
        # # generate input:
        # batch_size = gt.size(0)
        # self.rect = self.__rect()
        # top_l_x, top_l_y, rect_size, full = self.rect
        # mask = torch.zeros((batch_size, 1, full, full), dtype=torch.float32)
        # mask[:, :, top_l_y:top_l_y+rect_size, top_l_x:top_l_x+rect_size] = 1.0
        # img = gt*(1-mask) + mask*127.5
        
        # # set inputs
        # self.img = img.float().to(device)   # img is the masked img
        # self.gt  = gt.float().to(device)    # ground truth, clean img
        # self.mask = mask.float().to(device) # binary mask, 1 for mask 0 for not 
        # # TODO: 
        # # self.gener._cal_feat_masks(self.mask)   
        
        # # self.feat_mask_4 = self.refine_net.cal_feat_mask(3, self.mask)    # 32x32
        # Set the inputs
        self.img = img.float().to(device)   # img is the masked img
        self.gt  = gt.float().to(device)    # ground truth, clean img
        self.mask = mask.float().to(device) # binary mask, 1 for mask 0 for not
        self.feat_masks_list = self.gener.cal_feat_masks(self.mask)

    def __rect(self):
        low, high, full = self.args.sizes
        rect_size = np.random.choice(high-low+1) + low
        assert rect_size >= low and rect_size <= high, "value error"
        
        top_l_x = np.random.choice(full - rect_size)
        top_l_y = np.random.choice(full - rect_size)

        return [top_l_x, top_l_y, rect_size, full]
    
    # def perceptual_loss(self, A_feats, B_feats):
    #     assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    #     loss_value = 0.0
    #     for i in range(len(A_feats)):
    #         A_feat = A_feats[i]
    #         B_feat = B_feats[i]
    #         loss_value += torch.mean(torch.abs(A_feat - B_feat))
    #     return loss_value

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass
        
    def forward(self):
        # HRNET:
        # get the gt attention.
        self.gt_inputs = torch.cat((self.gt, self.mask), dim=1)
        self.inputs = torch.cat((self.img, self.mask), dim=1)
        # siamese training
        self.gt_pred_img, self.gt_score_list = self.gener(self.gt_inputs)  # gt
        # self.real_self_score = gt_out.score_self
        # self.real_cross_score = gt_out.score_cross
        # self.real_upx5 = gt_out.upx5
        
        # self.fake_B, self.fake_self_score, self.fake_cross_score = self.gener(self.img)     # fakeB
        self.pred_img, self.pred_score_list = self.gener(self.inputs)
        # self.fake_B = fake_out.final
        # self.fake_self_score = fake_out.score_self
        # self.fake_cross_score = fake_out.score_cross
        # self.fake_x4 = fake_out.x4
        # self.fake_upx5 = fake_out.upx5
        self.comp_img = (1 - self.mask)*self.img + self.mask*self.pred_img

    def optimize_parameters(self, update_d=False):
        self.train()
        self.forward()
        # Discriminator
        if update_d:
            self.optimizer_D.zero_grad()
            self.optimizer_F.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.optimizer_F.step()
        # Generator
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def backward_D(self):
        # fake B is the generated img 
        # img is the input masked img

        fake_AB = self.comp_img
        real_AB = self.gt
        self.gt_latent_fake = self.vgg(self.comp_img.clone().detach())
        self.gt_latent_real = self.vgg(self.gt.clone().detach())

        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        self.pred_fake_F = self.netF(self.gt_latent_fake['relu3_3'].detach())
        self.pred_real_F = self.netF(self.gt_latent_real['relu3_3'])
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F, self.pred_real_F, True)

        self.loss_D = self.loss_D_fake * 0.5  + self.loss_F_fake * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = self.comp_img
        pred_img = self.pred_img

        [self.gt_score_self_x5, self.gt_score_cross_x5, self.gt_score_self_x2, self.gt_score_cross_x2] = self.gt_score_list
        [self.pred_score_self_x5, self.pred_score_cross_x5, self.pred_score_self_x2, self.pred_score_cross_x2] = self.pred_score_list

        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(self.gt_latent_fake['relu3_3'])

        pred_real = self.netD(self.gt)
        pred_real_f = self.netF(self.gt_latent_real['relu3_3'])
        self.style_loss = self.StyleLoss(fake_AB, self.gt)
        self.perc_loss = self.PerceptualLoss(fake_AB, self.gt)
        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_f, False)
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.gt) * self.args.lambda_A
        # ymm - 9.18.2020 perceptual loss: attempt to get
        self.latent_fake = self.vgg(fake_AB.clone().detach())
        self.latent_real = self.vgg(self.gt.clone().detach())
        self.latent_pred = self.vgg(pred_img.clone().detach())
        # self.pecp_loss = self.perceptual_loss(self.latent_fake , self.latent_real) + self.perceptual_loss(self.latent_pred, self.latent_real)

        # attention Supervised
        self.loss_G_atten_x2 = self.criterionMSE(self.pred_score_self_x2, self.gt_score_self_x2) + self.criterionMSE(self.pred_score_cross_x2, self.gt_score_cross_x2)  # Attention loss. 
        self.loss_G_atten_x5 = self.criterionDiv(self.pred_score_self_x5, self.gt_score_self_x5) + self.criterionDiv(self.pred_score_cross_x5, self.gt_score_cross_x5)  # Attention loss. 
        # ymm - reconstruction loss [source: https://openaccess.thecvf.com/content_CVPR_2020/papers/Yi_Contextual_Residual_Aggregation_for_Ultra_High-Resolution_Image_Inpainting_CVPR_2020_paper.pdf]
        self.loss_recon = 6.0*self.criterionL1(self.comp_img*self.mask, self.gt*self.mask)/torch.mean(self.mask)   + 1.0*self.criterionL1(self.comp_img*(1-self.mask), self.gt*(1-self.mask))/torch.mean(1-self.mask) 
        # ymm - from pennet
        # self.pyramid_loss = 0 
        # if self.feats is not None:
        #     for _, f in enumerate(self.feats):
        #         self.pyramid_loss += self.criterionL1(f, F.interpolate(self.gt, size=f.size()[2:4], mode='bilinear', align_corners=True))

        self.loss_G = self.style_loss * 150 + self.loss_G_GAN * self.args.gan_weight + self.loss_recon*1 + 0.1 * self.perc_loss + 0.01 * self.loss_G_atten_x5 + 0.1 * self.loss_G_atten_x2#self.loss_G_L1

        # consistance loss:
        # self.consis_x4_loss = self.criterionL1(self.fake_x4, self.gt_latent_real.relu4_3)
        # self.loss_G += self.consis_x4_loss  # alpha ?

        # # siamese supervised. 
        # self.siamese_loss = self.criterionL1(self.fake_upx5, self.real_upx5)
        # self.loss_G += self.siamese_loss    # beta ? small one. 
        self.loss_G.backward()



    def get_current_visual(self):
        img = self.img.data
        real_B = self.gt.data
        comp_img = self.comp_img.data
        mask = self.mask.data

        return img, mask, comp_img, real_B

    def call_tfboard(self, writer, step):
        writer.add_scalar("G_GAN", self.loss_G_GAN.data.item(), step)
        writer.add_scalar("loss_recon", self.loss_recon.data.item(), step)
        writer.add_scalar("D", self.loss_D_fake.data.item(), step)
        writer.add_scalar("F", self.loss_F_fake.data.item(), step)

    def get_GAN_loss(self):
        return self.loss_G_GAN.data.item()

    def save_network(self, network, network_label, epoch_label, gpu_ids=[0]):
        if os.path.exists( self.save_dir ) is False:
            os.makedirs( self.save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def save(self, epoch):
        self.save_network(self.gener, 'PENnet', epoch)
        self.save_network(self.netD, 'D', epoch)
        self.save_network(self.netF, 'F', epoch)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr
    


    def test(self):
        self.eval()
        with torch.no_grad():
            self.inputs = torch.cat((self.img, self.mask), dim=1)
            self.pred_img, _ = self.gener(self.inputs)
            self.comp_img = (1 - self.mask)*self.img + self.mask*self.pred_img
