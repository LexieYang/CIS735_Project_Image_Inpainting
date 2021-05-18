import torch
import numpy as np
import skimage.io as io
import skimage.transform as trans
import os
from torch.utils.data import Dataset
import random
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import trange
# import cv2
from skimage.io import imread, imsave

random.seed(2333)


# class CelebA(Dataset):
#     def __init__(self, gt_root, img_root, mask_root, files, img_size=(256, 256), augmentation=False, transform=None, mask_transform=None):
#         self.files = files
#         self.img_size = img_size
#         self.gt_root = gt_root
#         self.img_root = img_root
#         self.mask_root = mask_root
#         self.transform = transform
#         self.mask_transform = mask_transform
#         self.augmentation = augmentation

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         """
#         The output image should including gt, g, and mask
        
#         """
#         filename = self.files[index]
#         filename = os.path.basename(filename)
#         img_path = os.path.join(self.img_root, filename)
#         gt_path = os.path.join(self.gt_root, filename.split('.')[0]+'.jpg')
#         mask_path = os.path.join(self.mask_root, filename)

#         gt_data = io.imread(gt_path)
#         img_data = io.imread(img_path)
#         mask_data = io.imread(mask_path, as_gray=True)
#         mask_data = np.expand_dims(mask_data, axis=2)
        
#         # mask_data = (gt_data != img_data) * 1.0  # generate the mask. 
#         # mask_data = np.sum(mask_data, axis=-1, keepdims=True) / 3
#         # io.imsave("./Debug/gt_data_before_resize.png", gt_data)
#         # io.imsave("./Debug/img_data_before_resize.png", img_data)
#         # io.imsave("./Debug/mask_data_before_resize.png", mask_data)

#         # resize
#         comp_data = np.concatenate([img_data, mask_data, gt_data], axis=-1)
#         comp_data = trans.resize(comp_data, self.img_size, order=0)

#         # img_data_, mask_data_, gt_data_ = np.split(comp_data, [3, 4], axis=-1)
#         # io.imsave("./Debug/gt_data_after_resize.png", gt_data_)
#         # io.imsave("./Debug/img_data_after_resize.png", img_data_)
#         # io.imsave("./Debug/mask_data_after_resize.png", mask_data_)

#         # transform:
#         if self.augmentation:
#             # rotation
#             degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
#             comp_data = trans.rotate(comp_data, degree)

#         # img_data_, mask_data_, gt_data_ = np.split(comp_data, [3, 4], axis=-1)
#         # io.imsave("./Debug/gt_data_after_augmentation.png", gt_data_)
#         # io.imsave("./Debug/img_data_after_augmentation.png", img_data_)
#         # io.imsave("./Debug/mask_data_after_augmentation.png", mask_data_)
#         # g_in, gt_data = np.split(comp_data, [4], axis=-1)
#         img_data, mask_data, gt_data = np.split(comp_data, [3, 4], axis=-1)

#         mask_data = self.mask_transform(mask_data)
#         gt_data = self.transform(gt_data)
#         img_data = self.transform(img_data)

#         # io.imsave("./Debug/g_in_after_augmentation_RGBA.png", img_data_ * mask_data_)

#         # g_in with 4 channels,  [RGBA]
#         # gt_data: 3 channels, [RGB]
#         # mask_data, 1 channels 
#         return img_data, gt_data, mask_data

class CelebA_Pre():
    def __init__(self, root, list_bbox, list_ldmk, list_parts):
        """
        `root`: dir, the path of images\\
        `list_*`, file path, csv files
        """
        assert os.path.isdir(root)
        self.root = root
        self.list_bbox = list_bbox
        self.list_ldmk = list_ldmk
        self.list_parts = list_parts

        self._init_DFs()


    def _init_DFs(self):
        """
        get the dataframes. 
        """
        self.DF_bbox = pd.read_csv(self.list_bbox, index_col=0)
        self.DF_ldmk = pd.read_csv(self.list_ldmk, index_col=0)
        self.DF_parts = pd.read_csv(self.list_parts)

    def get_parts(self, mode='eval'):
        """
        splitting all files into train/eval/test 
        Keep same partition as kaggle.
        ### Params
        `mode`: 'train', 'eval' or 'test'
        """
        mode_map = {
            'train':0,
            'eval':1,
            'test':2
        }

        ids = self.DF_parts[self.DF_parts['partition'] == mode_map[mode]]['image_id'].tolist()
        return ids

    def read_annote(self, id):
        """
        Get the annotation of given id. 
        ### Parames
        `id`: str, the id of target img, like '000003.jpg'
        
        ### Return
        `img_path`: str, the path for img,\\
        `bbox`: list of int, [x_1, y_1, width, height]\\
        `ldmk`: list of int, five point landmark, with shape (10, )\\
                lefteye_x  lefteye_y  righteye_x  righteye_y  nose_x  nose_y  leftmouth_x  leftmouth_y  rightmouth_x  rightmouth_y
        """
        img_path = os.path.join(self.root, id)
        bbox = self.DF_bbox.loc[id].to_numpy()
        ldmk = self.DF_ldmk.loc[id].to_numpy()

        return img_path, bbox, ldmk

    def legi_check(self, img_root, train_ids, eval_ids, test_ids):
        """
        return the legitimate ids. 
        """
        # train_ids
        legi_train = []
        legi_eval = []
        legi_test = []
        for id in train_ids:
            id = id.split('.')[0] + '.png'
            f = os.path.join(img_root, id)
            if os.path.isfile(f):
                legi_train.append(id)
            else:
                pass

        for id in eval_ids:
            id = id.split('.')[0] + '.png'
            f = os.path.join(img_root, id)
            if os.path.isfile(f):
                legi_eval.append(id)
        else:
            pass

        for id in test_ids:
            id = id.split('.')[0] + '.png'
            f = os.path.join(img_root, id)
            if os.path.isfile(f):
                legi_test.append(id)
        else:
            pass

        return legi_train, legi_eval, legi_test

        # chuck_size = total_len // num_folds
        # # cross validation
        # random.shuffle(legi_folds)
        # train_folds = []
        # eval_folds = [legi_folds[i*chuck_size:(i+1)*chuck_size] if (i+1) != num_folds else legi_folds[i*chuck_size:] for i in range(num_folds)]
        # legi_folds = set(legi_folds)
        # # XOR operation to get train_folds
        # for fold in eval_folds:
        #     set_eval = set(fold)
        #     train_folds.append(list(legi_folds.symmetric_difference(set_eval)))

        # return legi_folds, train_folds, eval_folds, legi_test

    def generate_list(self, save_dir, array, name):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        assert name.endswith(".txt")

        f1 = open(os.path.join(save_dir, name), 'w')
        for id in array:
            f1.write(os.path.basename(id) + '\n')
        f1.close()
        print("SAVED {}".format(os.path.join(save_dir, name)))

    def read_list(self, list_path, list_name, img_root ):
        file_name = os.path.join(list_path, list_name)
        assert os.path.isfile(file_name)
        assert os.path.isdir(img_root)

        legi_list = []

        f1 = open(file_name, 'r')
        lines = f1.readlines()
        f1.close()

        for l in lines:
            l = l.rstrip()
            f = os.path.join(img_root, l)
            legi_list.append(f)

        return legi_list

class CelebA(Dataset):
    def __init__(self, mode, gt_root, img_root, mask_root, file_list, sizes, mask_type, transform=None, mask_transform=None):
        """
        Lite way, 
        only for 256 256 input imgs. 
        rect: if True, __getitem__ return the rect value and gt
        """
        self.mode = mode
        self.files = file_list
        self.gt_root = gt_root
        self.img_root = img_root
        self.mask_root = mask_root
        # self.augmentation = augmentation
        self.transform = transform
        self.mask_transform = mask_transform
        self.sizes = sizes # saving the size information.
        # self.rect = rect   # 
        self.mask_type = mask_type
        # self.N_mask = len(mask_root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        if self.mask_type == "cnt_mask":
            gt_path = os.path.join(self.gt_root, filename)

            gt_data = io.imread(gt_path)
            gt_data = trans.resize(gt_data, self.sizes, order=0)
            if self.mode == "train":
                degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
                gt_data = trans.rotate(gt_data, degree)
            gt_data = self.transform(gt_data)
            mask_data = torch.rand(1, self.sizes[0], self.sizes[1])
            img_data = torch.rand(3, self.sizes[0], self.sizes[1])
            

        elif self.mask_type == "face_mask":  
# TODO: pay attention to the invalid images
            img_path = os.path.join(self.img_root, filename)
            
            gt_path = os.path.join(self.gt_root, filename.split('.')[0]+'.jpg')
            mask_path = os.path.join(self.mask_root, filename)

            gt_data = io.imread(gt_path)
            gt_data = trans.resize(gt_data, self.sizes, order=0)

            img_data = io.imread(img_path)
            img_data = trans.resize(img_data, self.sizes, order=0)

            mask_data = io.imread(mask_path, as_gray=True)
            mask_data = trans.resize(mask_data, self.sizes, order=0)
            mask_data = np.expand_dims(mask_data, axis=2)

            # transform:
            comp_data = np.concatenate([img_data, mask_data, gt_data], axis=-1)
            comp_data = trans.resize(comp_data, self.sizes, order=0)
            if self.mode == "train":
                degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
                comp_data = trans.rotate(comp_data, degree)

            img_data, mask_data, gt_data = np.split(comp_data, [3, 4], axis=-1)

            mask_data = self.mask_transform(mask_data)
            gt_data = self.transform(gt_data)
            img_data = self.transform(img_data)
        
        elif self.mask_type == "irr_mask":
            

            gt_path = os.path.join(self.gt_root, filename)

            gt_data = io.imread(gt_path)
            gt_data = trans.resize(gt_data, self.sizes, order=0)

            img_data = torch.rand(self.sizes[0], self.sizes[1], 3) # [256, 256, 3]


            mask_ratio_list = ["10_20", "20_30", "30_40", "40_50"]
            mask_ratio = random.choice(mask_ratio_list)
            self.mask_ratio_root = os.path.join(self.mask_root, mask_ratio)

            self.mask_paths = glob('{:s}/*.png'.format(self.mask_ratio_root))
            self.N_mask = len(self.mask_paths)

            mask_data = io.imread(self.mask_paths[random.randint(0, self.N_mask - 1)], as_gray=True)
            mask_data = trans.resize(mask_data, self.sizes)
            mask_data = np.expand_dims(mask_data, axis=2)
            # mask_data = self.mask_transform(mask_data)
            
            comp_data = np.concatenate([img_data, mask_data, gt_data], axis=-1)
            comp_data = trans.resize(comp_data, self.sizes, order=0)
            
            if self.mode == "train":
                degree = 90 * np.random.choice([0, 1, 2, 3], 1)[0]
                comp_data = trans.rotate(comp_data, degree)

            img_data, mask_data, gt_data = np.split(comp_data, [3, 4], axis=-1)

             # DEBUG
            # io.imsave("./Debug/gt_data_after_augmentation.png", gt_data)
            # io.imsave("./Debug/img_data_after_augmentation.png", img_data)
            # io.imsave("./Debug/mask_data_after_augmentation.png", mask_data)


            mask_data = self.mask_transform(mask_data)
            gt_data = self.transform(gt_data)
            img_data = self.transform(img_data)
        
        else:
            raise ValueError("Mask_type [%s] not recognized. Please choose among ['face_mask', 'cnt_mask', 'irr_mask']  " % self.mask_type)

       
        return gt_data, img_data, mask_data

    def __rect_mask(self):
        low, high, full = self.sizes
        rect_size = np.random.choice(high-low+1) + low
        assert rect_size >= low and rect_size <= high, "value error"

        top_l_x = np.random.choice(full - rect_size)
        top_l_y = np.random.choice(full - rect_size)

        mask = np.zeros((full, full, 1), dtype=float)
        mask[top_l_y:top_l_y+rect_size, top_l_x:top_l_x+rect_size, :] = 1.0

        return mask


class CelebA_RFR(torch.utils.data.Dataset):
    def __init__(self, file_list_path, mask_list_path, image_path, mask_path, mask_mode, target_size, augment=True, mode='training', mask_reverse = False):
        """
        Dataset prepare for RFR model. 
        """
        super(CelebA_RFR, self).__init__()
        self.augment = augment
        if mode == 'training':
            self.training = True
        else:
            self.training = False

        self.data = self.load_list(file_list_path, image_path)
        self.mask_data = self.load_list(mask_list_path, mask_path)

        self.target_size = target_size
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        img = imread(self.data[index])
        if self.training:
            img = self.resize(img)
        else:
            img = self.resize(img, True, True, True)
        # load mask
        mask = self.load_mask(img, index)
        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(mask)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        
        #external mask, random order
        if self.mask_type == 0:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, False)
            mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255
        #generate random mask
        if self.mask_type == 1:
            mask = 1 - generate_stroke_mask([256, 256])
            return (mask * 255).astype(np.uint8)
        
        #external mask, fixed order
        if self.mask_type == 2:
            mask_index = index % len(self.mask_data)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, False)
            mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

    def resize(self, img, aspect_ratio_kept = True, fixed_size = False, centerCrop=False):
        
        if aspect_ratio_kept:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)
            if fixed_size:
                if centerCrop:
                # center crop
                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]
                else:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                if side <= self.target_size:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
                else:
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
        # img = scipy.misc.imresize(img, [self.target_size, self.target_size])
        img = np.array(Image.fromarray(img).resize([self.target_size, self.target_size]))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_list(self, list_file, data_path):
        line = open(list_file,"r")
        lines = line.readlines()
        file_names = []
        for line in lines:
            line = line.rstrip()
            file_names.append(os.path.join(data_path, line))
        return file_names
##################### generate irregular masks ######################
def generate_stroke_mask(im_size, parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis = 2)
    return mask

# def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
#     mask = np.zeros((h, w, 1), np.float32)
#     numVertex = np.random.randint(maxVertex + 1)
#     startY = np.random.randint(h)
#     startX = np.random.randint(w)
#     brushWidth = 0
#     for i in range(numVertex):
#         angle = np.random.randint(maxAngle + 1)
#         angle = angle / 360.0 * 2 * np.pi
#         if i % 2 == 0:
#             angle = 2 * np.pi - angle
#         length = np.random.randint(maxLength + 1)
#         brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
#         nextY = startY + length * np.cos(angle)
#         nextX = startX + length * np.sin(angle)
#         nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
#         nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
#         cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
#         cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
#         startY, startX = nextY, nextX
#     cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
#     return mask

# def save_mask(mask_dir, amount = 7000):
#     path_10_20 = os.path.join(mask_dir, "10_20")
#     path_20_30 = os.path.join(mask_dir, "20_30")
#     path_30_40 = os.path.join(mask_dir, "30_40")
#     path_40_50 = os.path.join(mask_dir, "40_50")
#     # path_50_60 = os.path.join(mask_dir, "50_60")

#     if not os.path.isdir(path_10_20):
#         os.makedirs(path_10_20)
#     if not os.path.isdir(path_20_30):
#         os.makedirs(path_20_30)
#     if not os.path.isdir(path_30_40):
#         os.makedirs(path_30_40)
#     if not os.path.isdir(path_40_50):
#         os.makedirs(path_40_50)

#     cout_10 = 0
#     cout_20 = 0
#     cout_30 = 0
#     cout_40 = 0
#     # cout_50 = 0

#     for i in trange(100000):

#         parts= random.randint(2, 15)
#         mask = generate_stroke_mask([256, 256], parts=parts)   # one for hole, zero for bg
#         # rev_mask = 1-mask
#         ratio = np.sum(mask) / 255 / 255 / 3

#         mask = (mask*255).astype(np.uint8)

#         if ratio >=0.1 and ratio <= 0.2:
#             if cout_10 > amount-1:
#                 continue
#             file_name = '{:0>6}.png'.format(cout_10)
#             cout_10 += 1 
#             imsave(os.path.join(path_10_20, file_name), mask)
#         elif ratio >=0.2 and ratio <= 0.3:
#             if cout_20 > amount-1:
#                 continue
#             file_name = '{:0>6}.png'.format(cout_20)
#             cout_20 += 1 
#             imsave(os.path.join(path_20_30, file_name), mask)
#         elif ratio >=0.3 and ratio <= 0.4:
#             if cout_30 > amount-1:
#                 continue
#             file_name = '{:0>6}.png'.format(cout_30)
#             cout_30 += 1
#             imsave(os.path.join(path_30_40, file_name), mask)
#         elif ratio >=0.4 and ratio <= 0.5:
#             if cout_40 > amount-1:
#                 continue
#             file_name = '{:0>6}.png'.format(cout_40)
#             cout_40 += 1
#             imsave(os.path.join(path_40_50, file_name), mask)

#         if (cout_10 + cout_20 + cout_30 + cout_40) >= amount*4 - 4:
#             return

# # if __name__ == "__main__":
# #     print("generating irregular masks...")
# #     save_mask("/data1/minmin/CelebA/irregular_masks_ratio/")

