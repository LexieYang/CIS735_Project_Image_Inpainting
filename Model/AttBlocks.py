import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import namedtuple
import functools
from Model.spectral_norm import use_spectral_norm
import numpy as np
from Tools.Selfpatch import Selfpatch
class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes [8, 512, 32, 32]
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 3 * self.rate
        # raw_w is extracted for reconstruction
        # shape [8, 8192, 256]
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L] , k=4, shape=(8,512,4,4,256)
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k] = [1, 256, 512, 4, 4]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        # shape of f = [8, 512, 16, 16]
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L], ksize = 3, [8, 4608, 256]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L] = [8, 512, 3, 3, 256]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]]) # [1,1,32,32]
            if self.use_cuda:
                mask = mask.cuda()
        else:
            if int_fs[3] == 32:
                mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest')
            else:
                mask = F.interpolate(mask, scale_factor=1./(8*self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L] = [4, 1, 3, 3, 256]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]
        score = []
        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k = [1,1,3,3]
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3) [1,256,512,3,3]
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            
            yi = yi * mm  # [1, L, H, W]

            score.append(yi)
            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            # offsets.append(offset)
        score = torch.cat(score, dim=0)
        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs) # [1, 128, 64, 64]

        return score

class Cross_Attention(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(Cross_Attention, self).__init__()
        self.in_ch = in_ch
        self.inter_ch = inter_ch

        self.theta = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gate = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.W = nn.Conv2d(inter_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, mask):
        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        #top_l_x, top_l_y, rect_size, full = rect
        # slicing, The shape of x should be 4 dim. 
        f = x * mask
        f = f.contiguous().view(batch_size, self.in_ch, -1)
        g = x.clone()
        g = x * (1 - mask)
        g = g.contiguous().view(batch_size, self.in_ch, -1)

        K = self.theta(g).view(batch_size, self.inter_ch, -1)
        Q = self.phi(f).view(batch_size, self.inter_ch, -1)
        Q = Q.permute(0, 2, 1)

        score = torch.matmul(Q, K)
        score = F.softmax(score, dim=-1)

        V = self.gate(g).view(batch_size, self.inter_ch, -1)

        feat = torch.matmul(V, score.permute(0, 2, 1))
        feat = feat.view(batch_size, feat.size(1), height, width)

        # Residual
        target = self.W(feat) + x

        return target, score

class Self_Attention(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(Self_Attention, self).__init__()
        self.in_ch = in_ch
        self.inter_ch = inter_ch

        self.theta = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gate = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.W = nn.Conv2d(inter_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, mask):
        batch_size = x.size(0)
        height, width = x.size(2), x.size(3)
        #top_l_x, top_l_y, rect_size, full = rect
        # slicing, The shape of x should be 4 dim. 
        f = x * mask
        #f = x[:, :, top_l_y:top_l_y+rect_size, top_l_x:top_l_x+rect_size]
        f = f.contiguous().view(batch_size, self.in_ch, -1)

        K = self.theta(f).view(batch_size, self.inter_ch, -1)
        Q = self.phi(f).view(batch_size, self.inter_ch, -1)
        Q = Q.permute(0, 2, 1)

        score = torch.matmul(Q, K)
        score = F.softmax(score, dim=-1)

        V = self.gate(f).view(batch_size, self.inter_ch, -1)

        feat = torch.matmul(V, score.permute(0, 2, 1))
        feat = feat.view(batch_size, feat.size(1), height, width)
        # padding:
        # target = torch.zeros((batch_size, feat.size(1), full, full), device=feat.device)
        # target[:, :, top_l_y:top_l_y+rect_size, top_l_x:top_l_x+rect_size] = feat
        # Residual
        target = self.W(feat) + x

        return target, score
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)
  
class Equalization(nn.Module):
    def __init__(self, inner_nc):
        super(Equalization, self).__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        # gus = util.gussin(1.5).cuda()
        # self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(inner_nc, inner_nc//2, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(inner_nc//2),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            # nn.InstanceNorm2d(256),
            # nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)
        b, c, h, w = out_32.size()
        # gus = self.gus.float()
        # gus_out = out_32[0].expand(h * w, c, h, w)
        # gus_out = gus * gus_out
        # gus_out = torch.sum(gus_out, -1)
        # gus_out = torch.sum(gus_out, -1)
        # gus_out = gus_out.contiguous().view(b, c, h, w)
        csa2_in = F.sigmoid(out_32)
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1)
        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out = csa_a * csa2_fff
        out = torch.sum(out, -1)
        out = torch.sum(out, -1)
        out_csa = out.contiguous().view(b, c, h, w)
        # out_32 = out_csa], 1)
        out_32 = self.down(out_csa)
        return out_32
      
class DSA_Equal(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(DSA_Equal, self).__init__()
        self.self_atten = Self_Attention(in_ch, inter_ch)
        self.cross_atten = Cross_Attention(in_ch, inter_ch)
        # self.equal = Equalization(in_ch*2)
        self.W = nn.Sequential(
            nn.Conv2d(inter_ch*2, in_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(in_ch)
        )



    def forward(self, x, mask):
        self_feat, score_self = self.self_atten(x, mask)
        cross_feat, score_cross = self.cross_atten(x, mask)
        # feat = torch.cat((self_feat, cross_feat), 0)
        # feat = self_feat + cross_feat
        
        se_feat = torch.cat([self_feat, cross_feat], dim=1)
        out = self.W(se_feat)
        # equal_feat = self.equal(se_feat)
        # out = equal_feat
        return out, score_self, score_cross

class Cross_NoneLocal(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(Cross_NoneLocal, self).__init__()

        self.in_ch = in_ch
        self.inter_ch = inter_ch

        self.theta = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi   = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gate  = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False) 

    def forward(self, fg, bg):

        batch_size = fg.size(0)
        K = self.theta(bg).view(batch_size, self.inter_ch, -1) # [B, C, N']

        Q = self.phi(fg).view(batch_size, self.inter_ch, -1)   # [B, C, N]
        Q = Q.permute(0, 2, 1)                                 # [B, N, C]

        score = torch.matmul(Q, K)                             # [B, N, N']
        score = F.softmax(score, dim=-1)
        
        V = self.gate(bg).view(batch_size, self.inter_ch, -1)  # [B, C, N']
        
        feat = torch.matmul(V, score.permute(0, 2, 1))            # [B, C, N]
        
        return feat, score

class Self_NoneLocal(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(Self_NoneLocal, self).__init__()

        self.in_ch = in_ch
        self.inter_ch = inter_ch

        self.theta = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi   = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gate  = nn.Conv1d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False) 

    def forward(self, x):
        """
        x should be 2-d tensor. [batch, C, N] 
        """
        batch_size = x.size(0)
        K = self.theta(x).view(batch_size, self.inter_ch, -1)  # [B, C, N]
        Q = self.phi(x).view(batch_size, self.inter_ch, -1) # [B, C, N]
        Q = Q.permute(0, 2, 1)               # [B, N, C]
        
        score = torch.matmul(Q, K)                          # [B, N, N]
        score = F.softmax(score, dim=-1)                    # [B, N, N]

        V = self.gate(x).view(batch_size, self.inter_ch, -1)  # [B, C, N]

        feat = torch.matmul(V, score.permute(0, 2, 1))

        return feat, score

class DualNoneLocal(nn.Module):
    """
    https://arxiv.org/pdf/2003.13903v1.pdf
    """
    def __init__(self, in_ch, inter_ch):
        super(DualNoneLocal, self).__init__()
        self.self_nonlocal = Self_NoneLocal(in_ch, inter_ch)
        self.cross_nonlocal = Cross_NoneLocal(in_ch, inter_ch)

        self.gate = nn.Conv1d(inter_ch, in_ch, kernel_size=1, stride=1, bias=False)


    def forward(self, x, mask):
        batch_size = x.size(0)
        channel = x.size(1)
        h = x.size(2)
        fg = (x * mask).view(batch_size, channel, -1)
        bg = (x * (1-mask)).view(batch_size, channel, -1)

        feat_self, score_self = self.self_nonlocal(fg)
        feat_cross,score_cross = self.cross_nonlocal(fg, bg)

        # feat = feat_self + feat_cross
        # feat = self.gate

        feat = x + self.gate(feat_self + feat_cross).view(batch_size, -1, h, h)

        return feat, score_self, score_cross

class LocalBlock2D_Attention(nn.Module):
    """
    2D Local Block
    """
    def __init__(self, in_ch, inter_ch):
        super(LocalBlock2D_Attention, self).__init__()

        self.in_ch = in_ch
        self.inter_ch = inter_ch
        
        self.theta = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        self.phi   = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, mask):
        batch_size = x.size(0)
        flaten_mask = mask.view(batch_size, 1, -1)

        theta_x = self.theta(x).view(batch_size, self.inter_ch, -1)
        theta_x = theta_x * flaten_mask
        theta_x = theta_x.permute(0, 2, 1)                          # [batch, H*W, inter_ch]


        phi_x = self.phi(x).view(batch_size, self.inter_ch, -1)     # [batch, inter_ch, H*W]
        phi_x = phi_x * flaten_mask
        score = torch.matmul(theta_x, phi_x)                        # [batch, H*W, H*W]
        score = F.softmax(score, dim=-1)
        # score = score * flaten_mask
        return score

class NonLocal_cosis_attention(nn.Module):
    def __init__(self, in_ch, inter_ch, subsample=False):
        super(NonLocal_cosis_attention, self).__init__()

        self.in_ch = in_ch
        self.inter_ch = inter_ch
        
        self.theta = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        self.phi   = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        self.g     = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(inter_ch, in_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(in_ch)
        )

        
        if subsample:
            self.theta = nn.Sequential(self.theta, nn.MaxPool2d(kernel_size=(4, 4)))   # input 256, out 64
            self.phi   = nn.Sequential(self.phi,   nn.MaxPool2d(kernel_size=(4, 4)))
            self.g     = nn.Sequential(self.g,   nn.MaxPool2d(kernel_size=(4, 4)))

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_ch, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_ch, -1) 
        theta_x = theta_x.permute(0, 2, 1)                          # [batch, H*W, inter_ch]

        phi_x = self.phi(x).view(batch_size, self.inter_ch, -1)     # [batch, inter_ch, H*W]
        score = torch.matmul(theta_x, phi_x)                        # [batch, H*W, H*W]
        score = F.softmax(score, dim=-1)

        y = torch.matmul(score, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_ch, 64, 64)
        # y = F.upsample_nearest(y, scale_factor=4)
        y = F.interpolate(y, scale_factor=4, mode='bilinear')

        W_y = self.W(y)
        z = W_y + x

        return z

class NonLocalBlock2D_Attention(nn.Module):
    """
    2D None-Local Block
    """
    def __init__(self, in_ch, inter_ch, subsample=False):
        super(NonLocalBlock2D_Attention, self).__init__()

        self.in_ch = in_ch
        self.inter_ch = inter_ch
        
        self.theta = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        self.phi   = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        
        if subsample:
            self.theta = nn.Sequential(self.theta, nn.MaxPool2d(kernel_size=(4, 4)))   # input 256, out 64
            self.phi   = nn.Sequential(self.phi,   nn.MaxPool2d(kernel_size=(4, 4)))


    def forward(self, x):
        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_ch, -1) 
        theta_x = theta_x.permute(0, 2, 1)                          # [batch, H*W, inter_ch]

        phi_x = self.phi(x).view(batch_size, self.inter_ch, -1)     # [batch, inter_ch, H*W]
        score = torch.matmul(theta_x, phi_x)                        # [batch, H*W, H*W]
        score = F.softmax(score, dim=-1)
        return score

class GLnL_Module(nn.Module):
    """
    Gated Local non-Local attention module, 
    """
    def __init__(self, in_ch, inter_ch):
        super(GLnL_Module, self).__init__()

        self.in_ch = in_ch
        self.inter_ch = inter_ch

        self.g_non_local = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        self.g_local = nn.Conv2d(in_ch, inter_ch, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(inter_ch, in_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(in_ch)
        )
        self.W_local = nn.Sequential(
            nn.Conv2d(inter_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(inter_ch)
        )

        self.non_local = NonLocalBlock2D_Attention(in_ch, inter_ch)
        self.local     = LocalBlock2D_Attention(in_ch, inter_ch)   # calculated based on "mask"

    def forward(self, x, mask):
        batch_size = x.size(0)

        non_local_score = self.non_local(x)    # [batch, HW, HW]
        local_score     = self.local(x, mask)        # [batch, HW, HW]
        
        g_non_local_x = self.g_non_local(x).view(batch_size, self.inter_ch, -1)
        g_non_local_x = g_non_local_x.permute(0, 2, 1)             # [batch, HW, inter_ch]
        g_local_x = self.g_local(x).view(batch_size, self.inter_ch, -1)
        g_local_x = g_local_x.permute(0, 2, 1)


        non_local_y = torch.matmul(non_local_score, g_non_local_x)
        local_y     = torch.matmul(local_score, g_local_x)
        
        local_y = local_y.permute(0, 2, 1).contiguous()
        local_y = local_y.view(batch_size, self.inter_ch, x.size(2), x.size(3))
        local_y = self.W_local(local_y)
        # y = non_local_y + local_y
        # y = non_local_y
        non_local_y = non_local_y.permute(0, 2, 1).contiguous()
        non_local_y = non_local_y.view(batch_size, self.inter_ch, x.size(2), x.size(3))
        
        y = non_local_y + local_y
        W_y = self.W(y)

        z = W_y + x

        return z
