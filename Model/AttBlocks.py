import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import namedtuple
import functools
from Model.spectral_norm import use_spectral_norm
import numpy as np
# from Tools.Selfpatch import Selfpatch

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
