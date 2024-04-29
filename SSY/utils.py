import os
import numpy as np
from PIL import Image
import cv2
import torch
from datetime import datetime
import sys
import math
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import nibabel as nib

#--------------通用函数----------------

def np2torch(np_array):
    #SSY
    '''
    Convert numpy array to torch tensor
    input: numpy array
    output: torch tensor
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(np_array)


def check_args(args, rank=0):
    #https://github.com/ayu255/CMDSR/blob/main/tools/utils.py
    if rank == 0:
        # if args.use_docker:
        #     args.setting_file = args.checkpoint_dir + args.setting_file
        #     args.log_file = args.checkpoint_dir + args.log_file
        #     # os.makedirs(args.training_state, exist_ok=True)
        #     os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(args.setting_file, 'w') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')

    return args

#--------------医学图像相关函数----------------

def read_nii2np(path_nii):
    #SSY
    '''
    Read a nii file and return as a numpy array
    input: path to nii file
    output: numpy array
    '''
    np_arr = nib.load(path_nii).get_fdata()
    if np_arr.ndim == 2:
        np_arr = np.expand_dims(np_arr, axis=2)

    return np_arr
def save_nii(nii_path:str, np_array,source_nii_path=None):
    #SSY
    '''
    Save numpy file as nii file
    input:  
        nii_path:path to nii file
        np_array: nii file numpy array
        source_nii_path: source nii file path
    output: None
    '''
    # np_array = np_array.astype(np.float16)
    
    if source_nii_path!=None:
        source_nii = nib.load(source_nii_path)
        nii = nib.Nifti1Image(np_array, iffine=source_nii.affine,header=source_nii.header)
        nib.save(nii, nii_path)
    else:
        nii = nib.Nifti1Image(np_array, affine=np.eye(4))
        nib.save(nii, nii_path)


#-----------------自然图像相关函数----------------


def load_img(path):
    #SSY
    '''
    Load image from path
    input: path to image
    output: numpy array
    '''
    cv2_im = cv2.imread(path)
    cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    return cv2_im

def save_img(path, img):
    #SSY
    '''
    Save image to path
    input: path to save image, image numpy array
    output: None
    '''
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(path, img)
    

def bgr2ycbcr(img, only_y=True):
    #https://github.com/ayu255/CMDSR/blob/main/tools/utils.py
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    #https://github.com/ayu255/CMDSR/blob/main/tools/utils.py
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

if __name__ == '__main__':
    np_arr = np.zeros((256,256,3))
    ten = np2torch(np_arr)
    print(ten.shape)
    # save_img(r'C:\Users\92314\Desktop\test.jpg', load_img(r"C:\Users\92314\Desktop\img.jpg"))
    
    
    img = np.zeros((256,256,3)).astype(np.uint8)
    img[:,:,0] = 255
    # img[:,:,1] = 255
    # img[:,:,2] = 255
    save_img(r'C:\Users\92314\Desktop\test.jpg', img)
    
