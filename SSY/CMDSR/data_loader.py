from torch.utils.data import Dataset
import os
from copy import deepcopy
import random
import cv2
import numpy as np
import torch

from tools.data_preprocess import add_jpeg, add_noise, Gaussian_blur, fspecial_gaussian
import nibabel as nib


def Get_local_hq_paths(hq_file_path, shuffle=False):
    img_list = []    
    for img_name in os.listdir(hq_file_path):
        if img_name[-3:] == 'nii':
            img_list.append(img_name)
    if shuffle:
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        random.shuffle(img_list)
    hq_img_list = deepcopy(img_list) 
    return hq_img_list

def Np2Tensor(l):
    def _single_np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()

        return tensor


    return [_single_np2Tensor(_l) for _l in l]



class SRMultiGroupRandomTaskDataSet(Dataset):
    def __init__(self, is_train=True):
        super(SRMultiGroupRandomTaskDataSet, self).__init__()
        self.support_size = 40
        self.scale_factor = 4
        
        self.hr_path = r'C:\Users\92314\Desktop\DATA\HR'
        self.hr_img_list = Get_local_hq_paths(self.hr_path)
        
        self.blur_kernel_size = 15
        self.range_blur_sigma = 3.8
        self.low_blur_sigma = 0.2
        
        self.noise_level = 15
    def _load_file(self, idx):
        # idx = self._get_index(idx)
        
        hr = nib.load(os.path.join(self.hr_path, self.hr_img_list[idx])).get_fdata()
        filename = self.hr_img_list[idx]
        
        if hr.ndim == 2:
            hr = np.expand_dims(hr, axis=2)
        
        hr = hr-np.min(hr)/(np.max(hr)-np.min(hr))
        
        return hr , filename
        
        
    def random_degradation_param(self):
        blur_kernel_size, blur_sigma, noise_level = None, None, None
        deg_cof = "deg"
        #generate degradation cofficients
        blur_kernel_size = (self.blur_kernel_size, self.blur_kernel_size)
        rand_range = int(self.range_blur_sigma*10)
        blur_sigma = round(self.low_blur_sigma + random.randint(0, rand_range)/10, 2)
        # deg_cof += "_blur_{}".format(str(blur_sigma))

        noise_level = int(random.randint(0,self.noise_level))
        # deg_cof += "_noise_{}".format(str(noise_level))
        
        # if self.args.add_jpeg:
        #     jpeg_quality = int(random.randint(self.args.jpeg_low_quality,self.args.jpeg_low_quality+self.args.jpeg_quality_range))
        #     deg_cof += "_jpeg_{}".format(str(jpeg_quality))
        return blur_kernel_size, blur_sigma, noise_level
        
    def random_degradation_transfer(self, blur_kernel_size, blur_sigma, noise_level, hr_patch):
        # [h, w, c]
    
        lr_patch = cv2.GaussianBlur(hr_patch, blur_kernel_size, blur_sigma)
        lr_patch = cv2.resize(lr_patch,(48,48), interpolation=cv2.INTER_CUBIC) 

        gaussian = np.random.normal(0, noise_level/255, lr_patch.shape)
        lr_patch = lr_patch + gaussian
        
        if lr_patch.ndim == 2:
            lr_patch = np.expand_dims(lr_patch, axis=2)
        
        return lr_patch
    
    
    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        
        lr_patch_tensors, hr_patch_tesnors, filenames = [], [], []
        
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        
        blur_kernel_size, blur_sigma, noise_level = self.random_degradation_param()
        
        for i in range(self.support_size):
            hr_patch = self.get_single_patch(hr, 192)
            lr_patch = self.random_degradation_transfer(blur_kernel_size,blur_sigma,noise_level,hr_patch)
            
            lr_patch, hr_patch = Np2Tensor([lr_patch, hr_patch])
            
            lr_patch_tensors.append(lr_patch.unsqueeze(0))
            hr_patch_tesnors.append(hr_patch.unsqueeze(0))
            filenames.append(filename)

        lr_patch_tensors = torch.cat(lr_patch_tensors,0)
        hr_patch_tesnors = torch.cat(hr_patch_tesnors,0)  
        
        return lr_patch_tensors, hr_patch_tesnors, filenames

    def get_single_patch(self,img, patch_size=128):
        ih, iw = img.shape[:2]

        ix = random.randrange(0, iw - patch_size + 1)
        iy = random.randrange(0, ih - patch_size + 1)


        img_patch = img[iy:iy + patch_size, ix:ix + patch_size, :]

        return img_patch
    
    
    
    
    
    
    
    def __len__(self):
        return len(self.hr_img_list)  # * self.repeat


    def get_single_patch(self,img, patch_size=128):
        ih, iw = img.shape[:2]

        ix = random.randrange(0, iw - patch_size + 1)
        iy = random.randrange(0, ih - patch_size + 1)


        img_patch = img[iy:iy + patch_size, ix:ix + patch_size, :]

        return img_patch


class SRDataSet(Dataset):
    #验证集
    def __init__(self, is_train = False):
        super(SRDataSet, self).__init__()
        # self.scale_factor = int(args.scale_factor)
        self.support_size = 20
        self.scale_factor = 4
        self.hr_path = r'C:\Users\92314\Desktop\DATA\VAL\HR'
        self.hr_img_List = Get_local_hq_paths(self.hr_path)
        self.blur_kernel_size = 7
        self.blur_sigma = 2.6
    def __getitem__(self, idx):
        
        hr, lr,filename = self._load_file(idx)
        
        #ToTensor
        lr_patch, hr_patch = Np2Tensor([lr, hr])
        
        lr_support_patchs = []
        for i in range(self.support_size):
            lr_support_patch = self.get_single_patch(lr,48)
            lr_support_patch = Np2Tensor([lr_support_patch])
            lr_support_patch = lr_support_patch[0]
            lr_support_patchs.append(lr_support_patch.unsqueeze(0))
        lr_support_patchs = torch.cat(lr_support_patchs, dim=0)
        
        return lr_patch, lr_support_patchs, hr_patch, filename
        
    def __len__(self):
        return len(self.hr_img_List)

    
    def _load_file(self, idx):
        """输出
        hr[0,1]
        lr[0,1]
        """
        
        hr = nib.load(os.path.join(self.hr_path, self.hr_img_List[idx])).get_fdata()
        
        if hr.ndim == 2:
            hr = np.expand_dims(hr, axis=2)
        
        hr = hr-np.min(hr)/(np.max(hr)-np.min(hr))
        lr = self._create_degradation_lr(hr)
        filename = self.hr_img_List[idx]
        return hr,lr,filename
    
    def _create_degradation_lr(self,hr):
        #高斯模糊
        # blur_kernel_size = self.args.test_blur_kernel_size
        # blur_kernel_size = 7
        # # blur_sigma = self.args.test_blur_sigma
        # blur_sigma = 2.6
        kernel = fspecial_gaussian(self.blur_kernel_size, self.blur_sigma) #各向同性高斯模糊核
        
        lr = Gaussian_blur(hr, kernel, sf=self.scale_factor)
        
        # 加高斯噪声
        # noise_level = self.args.test_noise_level
        
        gaussian = np.random.normal(0, 0.05, lr.shape)
        lr = lr + gaussian
        
        return lr
    
    def get_single_patch(self,img, patch_size=128):
        ih, iw = img.shape[:2]

        ix = random.randrange(0, iw - patch_size + 1)
        iy = random.randrange(0, ih - patch_size + 1)


        img_patch = img[iy:iy + patch_size, ix:ix + patch_size, :]

        return img_patch


if __name__=="__main__":
    
    print("data_loader.py")
    
    from PIL import Image
    # val = SRDataSet()
    
    # lr_patch, lr_support_patchs, hr_patch, filename = val[0]
    
    # print(lr_patch.shape)
    # print(hr_patch.shape)
    # print(lr_support_patchs.shape)
    # print(filename)

    
    # img = Image.fromarray(lr_patch.numpy().transpose(1,2,0).astype(np.uint8).squeeze())
    # img.save('lr_patch.jpg')
    
    # img = Image.fromarray(hr_patch.numpy().transpose(1,2,0).astype(np.uint8).squeeze())
    # img.save('hr_patch.jpg')
    

    train = SRMultiGroupRandomTaskDataSet()
    
    lr_patch_tensors, hr_patch_tesnors, filenames = train[0]
    
    print(lr_patch_tensors.shape)
    print(hr_patch_tesnors.shape)
    print(filenames)