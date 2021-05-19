import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
import torchvision
from collections import namedtuple
import functools
from Tools.utils import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, same_padding
from Model.spectral_norm import use_spectral_norm
import numpy as np
# import util.util as util
# from Tools.Selfpatch import Selfpatch
from .AttBlocks import DSA_Equal



class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class Local_Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, device_ids=None):
        super(Local_Discriminator, self).__init__()
        self.in_dim = in_dim
        self.cnum = out_dim
        
        self.dis_conv_module = DisConvModule(in_dim, out_dim)
        self.linear = nn.Linear(out_dim*4, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

class DisConvModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.conv1 = dis_conv(in_dim, out_dim, 5, 2, 2)
        self.conv2 = dis_conv(out_dim, out_dim*2, 5, 2, 2)
        self.conv3 = dis_conv(out_dim * 2, out_dim * 4, 5, 2, 2)
        self.conv4 = dis_conv(out_dim * 4, out_dim * 4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x



def dis_conv(in_dim, out_dim, kernel_size=5, stride=2, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
        nn.LeakyReLU(0.2, inplace=True)
    )


class PFDiscriminator(nn.Module):
    def __init__(self):

       super(PFDiscriminator, self).__init__()


       self.model=nn.Sequential(
           nn.Conv2d(256, 512, kernel_size=4, stride=2,padding=1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(512, 512, kernel_size=4, stride=2,padding=1),
           nn.InstanceNorm2d(512),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(512, 512,kernel_size=4, stride=2,padding=1)

       )

    def forward(self, input):
        return self.model(input)





def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()


        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()


        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])


        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)


        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)


        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3':max_3,


            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,


            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out

class RRBlock_32(nn.Module):
    def __init__(self, nc):
        super(RRBlock_32, self).__init__()
        self.RB1 = _ResBlock_32(nc)
        # self.RB2 = _ResBlock_32()
        # self.RB3 = _ResBlock_32()

    def forward(self, input):
        out = self.RB1(input)
        # out = self.RB2(out)
        # out = self.RB3(out)
        return out

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def _activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
def _norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'in':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


class _ResBlock_32(nn.Module):
    def __init__(self, nc=512):
        super(_ResBlock_32, self).__init__()
        self.c1 = conv_layer(nc, nc // 4, 3, 1)
        self.d1 = conv_layer(nc // 4, nc // 4, 3, 1, 1)  # rate = 1
        self.d2 = conv_layer(nc // 4, nc // 4, 3, 1, 2)  # rate = 2
        self.d3 = conv_layer(nc // 4, nc // 4, 3, 1, 4)  # rate = 4
        self.d4 = conv_layer(nc // 4, nc // 4, 3, 1, 8)  # rate = 8
        self.act = _activation('relu')
        self.norm = _norm('in', nc)
        self.c2 = conv_layer(nc, nc, 3, 1)  # fusion

    def forward(self, x):
        output1 = self.act(self.norm(self.c1(x)))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = torch.cat([d1, add1, add2, add3], 1)
        output2 = self.c2(self.act(self.norm(combine)))
        output = x + self.norm(output2)
        return output





# source: https://arxiv.org/pdf/1904.07475.pdf
##################### ymm - reimplementation
class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()
  
  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).'% (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)

    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)


class Discriminator(BaseNetwork):
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      
      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),

      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)
    if init_weights:
      self.init_weights()


  def forward(self, x):
    x = self.encoder(x)
    label_x = self.classifier(x)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x
# Define the resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# define the Encoder unit
class UnetSkipConnectionEBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionEBlock, self).__init__()
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc, affine=True)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downrelu, downconv]
            model = down
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class UnetSkipConnectionDBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
        up = [uprelu, upconv, upnorm]

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Sep 24 2020 -- ymm -- different attention scheme
class MuliAttScheme(BaseNetwork):
    def __init__(self, input_nc, output_nc, ngf=64, res_num=4, norm_layer=nn.BatchNorm2d, use_dropout=True, init_weights=True):
        super(MuliAttScheme, self).__init__()
        # construct unet structure
        self.Encoder_1 = UnetSkipConnectionEBlock(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)
        self.Encoder_2 = UnetSkipConnectionEBlock(ngf, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Encoder_3 = UnetSkipConnectionEBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Encoder_4 = UnetSkipConnectionEBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Encoder_5 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Encoder_6 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True)

        # dilated conv: dilation = 1,2,4,8
         
        x4_blocks = []
        for _ in range(1):
            block = RRBlock_32(ngf*8)
            x4_blocks.append(block)
        self.dilated_x4 = nn.Sequential(*x4_blocks)


        x2_blocks = []
        for _ in range(1):
            block = RRBlock_32(ngf*2)
            x2_blocks.append(block)
        self.dilated_x2 = nn.Sequential(*x2_blocks)
        # attention module

        self.at_conv04 = DSA_Equal(ngf*8, ngf*8) # large-scale feature, like facial structure 

        self.at_conv02 = DSA_Equal(ngf*2, ngf*2)  # small-scale features, like hair
        # residual blocks
        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        # decoder
        self.Decoder_1 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout,
                                             innermost=True)
        self.Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_6 = UnetSkipConnectionDBlock(ngf * 2, output_nc, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)


        if init_weights:
            self.init_weights()

    def cal_feat_masks(self, mask):
        
        self.feat_masks_list = []
        # self.feat_masks_list.append(mask) # [256, 256]
        t_mask1 = mask
        t_mask1 = F.interpolate(t_mask1, scale_factor=np.power(0.5, 2), mode='nearest')
        self.feat_masks_list.append(t_mask1)
        
        t_mask2 = mask
        t_mask2 = F.interpolate(t_mask2, scale_factor=np.power(0.5, 4), mode='nearest')
        self.feat_masks_list.append(t_mask2)

    def forward(self, input):
        x1 = self.Encoder_1(input)
        x2 = self.Encoder_2(x1)
        x3 = self.Encoder_3(x2)
        x4 = self.Encoder_4(x3)
        x5 = self.Encoder_5(x4)
        x6 = self.Encoder_6(x5)
        x7 = self.middle(x6)
        # attention 
        dilated_x4 = self.dilated_x4(x4)
        dilated_x2 = self.dilated_x2(x2)
        feat_x4, score_self_x4, score_cross_x4 = self.at_conv04(dilated_x4, self.feat_masks_list[1])

        feat_x2, score_self_x2, score_cross_x2 = self.at_conv02(dilated_x2, self.feat_masks_list[0])

        # decoder
        y1 = self.Decoder_1(x7)
        y2 = self.Decoder_2(torch.cat([y1, x5], 1))
        y3 = self.Decoder_3(torch.cat([y2, feat_x4], 1))
        y4 = self.Decoder_4(torch.cat([y3, x3], 1))
        y5 = self.Decoder_5(torch.cat([y4, feat_x2], 1))
        y6 = self.Decoder_6(torch.cat([y5, x1], 1))
        out = y6

        return out, [score_self_x4, score_cross_x4, score_self_x2, score_cross_x2]


