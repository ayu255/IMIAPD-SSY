from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as ttf
import cv2
import nibabel as nib
import numpy as np
import os
import pydicom as dicom
from data_generation import apply_degradation, get_LR_image
from glob import glob

import utils








class BaseDataSet(Dataset):
    # https://github.com/Samiran-Dey/BliMSR/blob/main/dataset_random.py
    # Data Building
    def __init__(self,inputs,transform=None):
        super().__init__()

        self.x = inputs
        
        self.n_samples = len(inputs)
        self.transform = transform
    
    # get an Item
    def __getitem__(self,index):
        inputs = self.x[index]
        hr = self.transform(inputs) 
        lr = get_LR_image(inputs)
        lr = self.transform(lr)
        return hr, lr
    
    def __len__(self):
        return self.n_samples



def read_nii(image_path):
    #SSY
    #读取nii文件
    image_paths = glob(image_path + '/*.nii')
    print (f'Total of {len(image_paths)} NII images.' )
    slices = [utils.read_nii2np(path) for path in image_paths]

    images=[]
    for img in slices:
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images.append(norm_img)
    
    return images

def create_nii_dataset(root_path, batch_size=1):
    #SSY
    #输入nii文件路径，路径下全是nii文件，返回一个最基本的数据集
    Nii_images=read_nii(root_path)
    transform_hr = ttf.ToTensor()
    data = BaseDataSet(Nii_images,transform_hr)
    return data


def read_dicom(image_path):
    #https://github.com/Samiran-Dey/BliMSR/blob/main/dataset_random.py
    #读取dcm文件 
  print (f'Total of {len(image_path)} DICOM images.' )
  slices = [dicom.read_file(path) for path in image_path]
  images=[]
  for ct in slices:
      img=ct.pixel_array
      norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      images.append(norm_img)
  return images

def create_dcm_dataset(root_path, batch_size=1):
    #https://github.com/Samiran-Dey/BliMSR/blob/main/dataset_random.py
    #获得dcm文件路径
  CT_path=[]
  for path in os.listdir(root_path):
    folders = os.listdir(root_path+path)
    for f in folders:
      subdir = os.listdir(root_path + path + '/' + f)
      for impath in subdir:
        image_path = root_path + path + '/' + f + '/' + impath
        data_paths = glob(image_path + '/*.dcm')
        data_paths.sort()
        CT_path += data_paths
    #读取文件
  CT_images=read_dicom(CT_path)
    #创建数据集
  transform_hr = ttf.ToTensor()
  data = BaseDataSet(CT_images,transform_hr)
  print('Loading data ... ')
  rn_data=[]
  for i in range(data.n_samples):
    rn_data.append(data.__getitem__(i))
  return rn_data



if __name__ == '__main__':
    dataset = create_nii_dataset(r'C:\Users\92314\Desktop\DATA\VAL\HR')
    
    hr,lr = dataset[10]
    
    from utils import save_nii
    
    print(hr.shape,lr.shape)
    print(hr.numpy().shape)
    save_nii('hr.nii', hr.numpy().squeeze())
    save_nii('lr.nii', lr.numpy().squeeze())