import torch
import torchvision
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os


class SR_Dataloader(Dataset):
    def __init__(self,hr_dir,crop_size,scale_factor):
        self.hr_dir=hr_dir
        self.imgs=sorted([os.path.join(self.hr_dir,fname) for fname in os.listdir(self.hr_dir)])
        self.hr_transform=transforms.Compose([transforms.CenterCrop(int(crop_size)),transforms.ToTensor()])
        self.lr_transform=transforms.Compose([transforms.CenterCrop(int(crop_size)),transforms.Resize(crop_size//scale_factor,interpolation=Image.BICUBIC),transforms.ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        high_res_image=self.hr_transform(Image.open(self.imgs[idx]))
        # high_res_image=0.2989*high_res_image[0]+0.5870*high_res_image[1]+0.1140*high_res_image[2]

        low_res_image=self.lr_transform(Image.open(self.imgs[idx]))
        # low_res_image=0.2989*low_res_image[0]+0.5870*low_res_image[1]+0.1140*low_res_image[2]

        return {"high_res":high_res_image,"low_res":low_res_image}
