import pydicom as dicom
from glob import glob
import cv2
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as ttf
from degradation import random_mixed_kernels, random_add_gaussian_noise, random_add_poisson_noise, random_add_jpg_compression, circular_lowpass_kernel


from utils import load_img,save_img

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_random_parameters():
  #from ssy
    '''生成随机的退化参数'''
    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    kernel=random_mixed_kernels(kernel_list, kernel_prob, kernel_size=21)
    
    interpolation_list = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR]   #随机选一个插值方法
    interpolation_prob = [0.6,0.2,0.2]
    interpolation = random.choices(interpolation_list, interpolation_prob)[0]
    
    noise_list = ['gaussian', 'poisson']
    noise_prob = [0.5,0.5]
    noise = random.choices(noise_list, noise_prob)[0]
    
    return kernel, interpolation, noise
  
def apply_fixed_degradation(img,kernel,interpolation,noise,scale = 4):
  #from ssy
    '''输入卷积核，插值方法，噪声，然后执行退化。特点是他不是随机的，而是固定的'''
    img = cv2.filter2D(src=img, kernel=kernel, ddepth=-1)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    size=img.shape  #正方形
    # print(size)   #下面这个是反过来的
    img = cv2.resize(img, (size[1]//scale, size[0]//scale), interpolation = interpolation) #一次缩小2倍
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    if noise == 'gaussian':
        img = random_add_gaussian_noise(img)
    else :
        img = random_add_poisson_noise(img)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img = random_add_jpg_compression(img)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  
  
    sinc_kernel = circular_lowpass_kernel(cutoff=np.random.uniform(np.pi / 3, np.pi), kernel_size=11)  #用低通滤波来模拟振铃和过冲
    img = cv2.filter2D(src=img, kernel=sinc_kernel, ddepth=-1)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img[img<=0]=0
    if img.ndim == 2:              #因为jprg压缩的最好一个     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 会把灰度图像的通道维度直接压缩
        img = np.expand_dims(img,axis = 2)
    
    return img

def apply_degradation(img, semi=False):  #第一阶 模糊，下采，加噪，压缩 第二阶 只 下采和压缩
  #https://github.com/Samiran-Dey/BliMSR/blob/main/dataset_random.py
  '''完全随机的退化,semi 是第几阶段的退化，如果是第二阶段，只有下采和压缩'''
  if not semi:
    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    kernel=random_mixed_kernels(kernel_list, kernel_prob, kernel_size=21)
    img = cv2.filter2D(src=img, kernel=kernel, ddepth=-1)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  interpolation_list = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR]   #随机选一个插值方法
  interpolation_prob = [0.6,0.2,0.2]
  interpolation = random.choices(interpolation_list, interpolation_prob)[0]
  size=img.shape  #正方形
  img = cv2.resize(img, (size[1]//2, size[0]//2), interpolation = interpolation) #一次缩小2倍
  if not semi:
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    noise_list = ['gaussian', 'poisson']
    noise_prob = [0.5,0.5]
    noise = random.choices(noise_list, noise_prob)[0]
    if noise == 'gaussian':
      img = random_add_gaussian_noise(img)
    else :
      img = random_add_poisson_noise(img)
  img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  img = random_add_jpg_compression(img)
  img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  
  if img.ndim == 2:              #因为jprg压缩的最好一个     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 会把灰度图像的通道维度直接压缩
    img = np.expand_dims(img,axis = 2)
    
  return img

def get_LR_image(img):  #随机生成一个低分辨率图像
  #https://github.com/Samiran-Dey/BliMSR/blob/main/dataset_random.py
  img = apply_degradation(img)
  img = apply_degradation(img, True)
  sinc_kernel = circular_lowpass_kernel(cutoff=np.random.uniform(np.pi / 3, np.pi), kernel_size=11)  #用低通滤波来模拟振铃和过冲
  img = cv2.filter2D(src=img, kernel=sinc_kernel, ddepth=-1)
  img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  img[img<=0]=0
  return img

if __name__ == '__main__':
    # img = np.random.rand(512, 1024, 1).astype(np.float32)
    img = load_img(r"C:\Users\92314\Desktop\img.jpg")
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    
    k,c,n = generate_random_parameters()
    img = apply_fixed_degradation(img,k,c,n)
    
    # # img = apply_degradation(img)
    # # img = apply_degradation(img)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  #会自动把通道维度压缩
    save_img('test.jpg', img)
    print(img.shape)