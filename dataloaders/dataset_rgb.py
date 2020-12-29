import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, split, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        assert split in ['train', 'test'], 'split must be "train"|"test"'  # train, val

        self.split     = split

        self.shape_images = []
        self.blur_images = []

        if self.split == 'train':
            data_dir = '/home2/zengwh/DeblurGANv2/GOPRO_Large/train/'
        else:
            data_dir = '/home2/zengwh/DeblurGANv2/GOPRO_Large/test/'
        for train_dir in os.listdir(data_dir):
            for image_name in os.listdir(os.path.join(data_dir,train_dir,'sharp')):
                self.shape_images.append(os.path.join(data_dir,train_dir,'sharp',image_name))
                self.blur_images.append(os.path.join(data_dir,train_dir,'blur',image_name))

        self.img_options=img_options

        self.tar_size = len(self.shape_images) # get the size of target

    def __len__(self):
        return self.tar_size
        #return  12

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.shape_images[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.blur_images[tar_index])))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.shape_images[tar_index])[-1]
        noisy_filename = os.path.split(self.blur_images[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        #apply_trans = transforms_aug[random.getrandbits(3)]

        #clean = getattr(augment, apply_trans)(clean)
        #noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


class DataLoaderTrain_fusion(Dataset):
    def __init__(self, split, img_options=None, target_transform=None):
        super(DataLoaderTrain_fusion, self).__init__()

        self.target_transform = target_transform

        assert split in ['train', 'test'], 'split must be "train"|"test"'  # train, val

        self.split = split

        self.shape_images = []
        self.blur_images = []
        self.dark_images = []

        if self.split == 'train':
            data_dir = '/home2/zengwh/deblur/defocus-deblurring-dual-pixel/datasets/source/train/'
        else:
            data_dir = '/home2/zengwh/deblur/defocus-deblurring-dual-pixel/datasets/source/train/'
        for train_dir in os.listdir(os.path.join(data_dir,'shape')):
            for image_name in os.listdir(os.path.join(data_dir,'shape',train_dir)):
                self.shape_images.append(os.path.join(data_dir,'shape',train_dir,image_name))
                self.blur_images.append(os.path.join(data_dir,'blur',train_dir,image_name))
                self.dark_images.append(os.path.join(data_dir, 'dark', train_dir, image_name))
        self.img_options = img_options

        self.tar_size = len(self.shape_images)  # get the size of target

    def __len__(self):
        return self.tar_size
        # return  12

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.shape_images[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.blur_images[tar_index])))
        dark = torch.from_numpy(np.float32(load_img(self.dark_images[tar_index])))


        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)
        dark = dark.permute(2, 0, 1)

        clean_filename = os.path.split(self.shape_images[tar_index])[-1]
        noisy_filename = os.path.split(self.blur_images[tar_index])[-1]
        dark_filename = os.path.split(self.dark_images[tar_index])[-1]

        # Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        dark = dark[:, r:r + ps, c:c + ps]
        # apply_trans = transforms_aug[random.getrandbits(3)]

        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, dark, clean_filename, noisy_filename, dark_filename
##################################################################################################
class DataLoaderVal_fusion(Dataset):
    def __init__(self, split, target_transform=None):
        super(DataLoaderVal_fusion, self).__init__()

        self.target_transform = target_transform

        assert split in ['train', 'test'], 'split must be "train"|"test"'  # train, val

        self.split     = split

        self.shape_images = []
        self.blur_images = []
        self.dark_images = []

        if self.split == 'train':
            data_dir = '/home2/zengwh/deblur/defocus-deblurring-dual-pixel/datasets/source/train/'
        else:
            data_dir = '/home2/zengwh/deblur/defocus-deblurring-dual-pixel/datasets/source/train/'
        for train_dir in os.listdir(os.path.join(data_dir,'shape')):
            for image_name in os.listdir(os.path.join(data_dir,'shape',train_dir)):
                self.shape_images.append(os.path.join(data_dir,'shape',train_dir,image_name))
                self.blur_images.append(os.path.join(data_dir,'blur',train_dir,image_name))
                self.dark_images.append(os.path.join(data_dir, 'dark', train_dir, image_name))

        self.tar_size = len(self.shape_images)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.shape_images[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.blur_images[tar_index])))
        dark = torch.from_numpy(np.float32(load_img(self.dark_images[tar_index])))


        clean_filename = os.path.split(self.shape_images[tar_index])[-1]
        noisy_filename = os.path.split(self.blur_images[tar_index])[-1]
        dark_filename = os.path.split(self.dark_images[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        dark = dark.permute(2,0,1)

        return clean, noisy,dark,  clean_filename, noisy_filename, dark_filename


class DataLoaderVal(Dataset):
    def __init__(self, split, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        assert split in ['train', 'test'], 'split must be "train"|"test"'  # train, val

        self.split = split

        self.shape_images = []
        self.blur_images = []

        if self.split == 'train':
            data_dir = '/home2/zengwh/DeblurGANv2/GOPRO_Large/train/'
        else:
            data_dir = '/home2/zengwh/DeblurGANv2/GOPRO_Large/test/'
        for train_dir in os.listdir(data_dir):
            for image_name in os.listdir(os.path.join(data_dir, train_dir, 'sharp')):
                self.shape_images.append(os.path.join(data_dir, train_dir, 'sharp', image_name))
                self.blur_images.append(os.path.join(data_dir, train_dir, 'blur', image_name))

        self.tar_size = len(self.shape_images)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.shape_images[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.blur_images[tar_index])))

        clean_filename = os.path.split(self.shape_images[tar_index])[-1]
        noisy_filename = os.path.split(self.blur_images[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename

class DataLoaderVal_deblur(Dataset):
    def __init__(self, split, target_transform=None):
        super(DataLoaderVal_deblur, self).__init__()

        self.target_transform = target_transform

        assert split in ['train', 'test'], 'split must be "train"|"test"'  # train, val

        self.split = split

        self.shape_images = []
        self.blur_images = []

        if self.split == 'train':
            data_dir = '/home2/zengwh/DeblurGANv2/GOPRO_Large/train/'
        else:
            data_dir = '/home2/zengwh/DeblurGANv2/GOPRO_Large/test/'
        for train_dir in os.listdir(data_dir):
            for image_name in os.listdir(os.path.join(data_dir, train_dir, 'sharp')):
                self.shape_images.append(os.path.join(data_dir, train_dir, 'sharp', image_name))
                self.blur_images.append(os.path.join(data_dir, train_dir, 'blur', image_name))

        self.tar_size = len(self.shape_images)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.shape_images[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.blur_images[tar_index])))

        clean_filename = self.shape_images[tar_index]
        noisy_filename = self.blur_images[tar_index]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename



class DataLoaderVal_deblur_fusion(Dataset):
    def __init__(self, split, target_transform=None):
        super(DataLoaderVal_deblur_fusion, self).__init__()

        self.target_transform = target_transform

        assert split in ['train', 'test'], 'split must be "train"|"test"'  # train, val

        self.split = split

        self.shape_images = []
        self.blur_images = []
        self.dark_images = []
        if self.split == 'train':
            data_dir = '/home2/zengwh/deblur/defocus-deblurring-dual-pixel/datasets/source/train/'
        else:
            data_dir = '/home2/zengwh/deblur/defocus-deblurring-dual-pixel/datasets/source/test/'
        for train_dir in os.listdir(os.path.join(data_dir,'shape')):
            for image_name in os.listdir(os.path.join(data_dir,'shape',train_dir)):
                self.shape_images.append(os.path.join(data_dir,'shape',train_dir,image_name))
                self.blur_images.append(os.path.join(data_dir,'blur',train_dir,image_name))
                self.dark_images.append(os.path.join(data_dir,'dark', train_dir, image_name))

        self.tar_size = len(self.shape_images)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.shape_images[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.blur_images[tar_index])))
        dark = torch.from_numpy(np.float32(load_img(self.dark_images[tar_index])))


        clean_filename =self.shape_images[tar_index]
        noisy_filename = self.blur_images[tar_index]
        dark_filename = self.dark_images[tar_index]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        dark = dark.permute(2,0,1)

        return clean, noisy,dark,  clean_filename, noisy_filename, dark_filename


##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2,0,1)

        return noisy, noisy_filename


##################################################################################################

class DataLoaderTestSR(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTestSR, self).__init__()

        self.target_transform = target_transform

        LR_files = sorted(os.listdir(os.path.join(rgb_dir)))


        self.LR_filenames = [os.path.join(rgb_dir, x) for x in LR_files if is_png_file(x)]
        

        self.tar_size = len(self.LR_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        LR = torch.from_numpy(np.float32(load_img(self.LR_filenames[tar_index])))
                
        LR_filename = os.path.split(self.LR_filenames[tar_index])[-1]

        LR = LR.permute(2,0,1)

        return LR, LR_filename
