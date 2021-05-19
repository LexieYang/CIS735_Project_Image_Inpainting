import os 
import json

class Base_Config():
    def __init__(self):
        self.seed = 2333
        self.img_dir  = "/data1/minmin/CelebA_With_Masks/"
        self.mask_dir  = "/data1/minmin/Binary_Masks/"
        self.data_dir = "/data1/minmin/CelebA/img_align_celeba/"
        self.ckpt_dir = "./model_weights/"
        self.dataset = "CelebA"
        self.visual_dir = "./Visual/"
        self.summary_dir = "./tfboards/"
        self.bbox_dir = "./Dataset/CelebA/list_bbox_celeba.csv"
        self.ldmk_dir = "./Dataset/CelebA/list_landmarks_align_celeba.csv"
        self.parts_dir = "./Dataset/CelebA/list_eval_partition.csv"
        self.json_path = './Experiments/logs/'
        self.device = 'cuda'
        # self.weights_

        # place holder:
        self.model = 'MODEL'
        self.name = "NAME"
        
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.epoch_count=0

        self.beta1 = 0.5
        self.lambda_A = 100
        self.gan_weight = 0.1

        self.visual_freq = 15000     # default 15000
        self.writer_freq = 500
        self.save_epoch_freq = 4
        self.ngpu  = 1
        self.mask_type =  "face_mask"    #["face_mask", "cnt_mask","irr_mask"]  # three types of masks, including face masks,\

    def save_config(self):
        with open(os.path.join(self.json_path, "{}-{}-{}".format(self.dataset, self.model, self.name)), "w+") as f:
            json.dump(self.__dict__, f, indent=4)


class ISSUE_11_EXP1(Base_Config):
    def __init__(self, mode="train"):
        super(ISSUE_11_EXP1, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 4   # around 4
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = (256, 256) 

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = ''        # loading from which epoch
        self.model = "PSA"  # abbreviation for Gated Local non Local Attention Net
        self.name = "ISSUE_11_EXP1"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators

class ISSUE_8_EXP2(Base_Config):
    """
    This EXP only work for branch GLnL_EXP2
    
    """
    def __init__(self, mode="train"):
        super(ISSUE_8_EXP2, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 8   # around 16
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of Unet (Coarse and Refine)

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = '48'        # loading from which epoch
        self.model = "GLnLAttenNet"  # abbreviation for Gated Local non Local Attention Net
        self.name = "ISSUE_8_EXP2_4" 
        self.mode = mode          # which mode
        self.summary = True
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 1e-3


class ISSUE_11_EXP3(Base_Config):
    def __init__(self, mode="train"):
        super(ISSUE_11_EXP3, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 8   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = ''        # loading from which epoch
        self.model = "Unet"  # [Unet or PSA]
        self.name = "ISSUE_11_EXP3"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators

class ISSUE_14_EXP1(Base_Config):
    def __init__(self, mode="train"):
        super(ISSUE_14_EXP1, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 14   # around 14
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 32  # The base dim  

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = ''        # loading from which epoch
        self.model = "Unet"  # [Unet or PSA]
        self.name = "ISSUE_14_EXP1"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators
        


class ISSUE_15_EXP4(Base_Config):
    """
    Compared with EXP1:

        # Add one more attention layer: dual3
        # no SE layer
        # no attention loss
        # perceptual loss + style loss
    """
    def __init__(self, mode="train"):
        super(ISSUE_15_EXP4, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 4   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = True          # loading model from ckpt. 
        self.which_epoch = '116'        # loading from which epoch
        self.model = "pcstyle_net"  # [Unet or PSA]
        self.name = "ISSUE_15_EXP4"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators




class ISSUE_16_EXP1(Base_Config):
    """
    add equalization layer
    """
    def __init__(self, mode="train"):
        super(ISSUE_16_EXP1, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 1   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = '32'        # loading from which epoch
        self.model = "equalnet"  # [Unet or PSA]
        self.name = "ISSUE_16_EXP1"
        self.mode = mode          # which mode
        self.summary = False         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators


class ISSUE_16_EXP2(Base_Config):
    """
    add equalization layer ===> add dilation conv
    """
    def __init__(self, mode="train"):
        super(ISSUE_16_EXP2, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 1   # around 8
        self.in_ch = 4   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = '52'        # loading from which epoch
        self.model = "equal_dilated_net"  # [Unet or PSA]
        self.name = "ISSUE_16_EXP2"
        self.mode = mode          # which mode
        self.summary = False         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators





class ISSUE_16_EXP3(Base_Config):
    """
    add equalization layer ===> add dilation conv ==> add resblocks
    """
    def __init__(self, mode="train"):
        super(ISSUE_16_EXP3, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 1   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = '56'        # loading from which epoch
        self.model = "equal_dilated_res_net"  # [Unet or PSA]
        self.name = "ISSUE_16_EXP3"
        self.mode = mode          # which mode
        self.summary = False         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators





class ISSUE_16_EXP4(Base_Config):
    """
    add equalization layer ===> add dilation conv ==> add resblocks ==> remove equalization layer
    """
    def __init__(self, mode="train"):
        super(ISSUE_16_EXP4, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 4   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = '96'        # loading from which epoch
        self.model = "dilated_res_net"  # [Unet or PSA]
        self.name = "ISSUE_16_EXP4"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators






class ISSUE_16_EXP6(Base_Config):
    """
    add equalization layer ===> add dilation conv ==> add resblocks ==> remove equalization layer
    """
    def __init__(self, mode="train"):
        super(ISSUE_16_EXP6, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 4   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = [94, 128, 256] # min, max value of rect. and size of img

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = ''        # loading from which epoch
        self.model = "dilated_res_net2"  # [Unet or PSA]
        self.name = "ISSUE_16_EXP6"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators



class ISSUE_16_EXP7(Base_Config):
    """
    add equalization layer ===> add dilation conv ==> add resblocks ==> remove equalization layer
    """
    def __init__(self, mode="train"):
        super(ISSUE_16_EXP7, self).__init__()
        self.niter=20        # final num of epochs equal to (niter + niter_decay + 1)
        self.niter_decay=100
        self.batch_size = 1   # around 8
        self.in_ch = 3   # the input channel of coarse stage
        self.out_ch = 3
        self.base_dim = 64  # The base dim of discriminator 

        self.sizes = (256, 256) 

        self.resume = False          # loading model from ckpt. 
        self.which_epoch = '40'        # loading from which epoch
        self.model = "dilated_res_net"  # [Unet or PSA]
        self.name = "ISSUE_16_EXP7"
        self.mode = mode          # which mode
        self.summary = True         # flg for tfboards
        self.summary_register = ['train']
        self.teacher_forcing_ratio = 1.0 # 

        self.lr = 2e-4
        # self.wait = 10000          # if after |wait| images,  GANloss still not decreasing, then stop the update the discriminators



