"""This file defines the data generator that is used to load the data during training. Here, on-the-fly under-sampling
and contrast inversion is performed.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset


class ReImgChannels(object):
    def __call__(self, img):
        """
        Convert a complex tensor into a 2 channel real/imaginary tensor
        Args:
            img (torch tensor): Complex valued torch tensor
        """
        c, h, w = img.shape
        empty_img = torch.empty((c * 2, h, w), dtype=torch.float64)
        re_img, im_img = torch.real(img), torch.imag(img)
        empty_img[::2, :, :] = re_img
        empty_img[1::2, :, :] = im_img
        return empty_img


class SliceSmapDataset(Dataset):
    def __init__(self, data_df, split, smaps, us_masks, target_type, coils, data_transforms=None,
                 target_transforms=None, invert='yes') -> None:
        super().__init__()
        '''
        Dataset class for 2D slices 
        Args:
            data_df (DataFrame): Contains slice paths, patient ids, slice numbers, and splits
            split (str): One of train, val, test
            smaps (List): Contains paths to the various sensitivity maps
            us_masks (List): Contains paths to the various undersampling masks
            target_type (str): ESPiRIT or NLINV used for recosntructing the target
            coils (int): how many coils in this multi-coil image
            data_transforms(callable, optional): Optional composition of tranforms for the input data
            target_transforms(callable, optional): Optional composition of transforms for the target data
        '''
        if target_type == 'nlinv':
            self.file_paths = data_df.loc[data_df['split'] == split, 'nlinv_path'].tolist()
        else:
            self.file_paths = data_df.loc[data_df['split'] == split, 'espirit_path'].tolist()
            # This is basically a csv with espirit_path, split, nlinv path
        self.smaps = smaps

        # self.file_paths = random.sample(self.file_paths, 50)
        self.us_masks = us_masks
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.coils = coils
        self.invert = invert

        if self.invert != 'no':
            self.matter_paths = data_df.loc[data_df['split'] == split, 'matter'].tolist()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        # print ('file and idx len', len(self.file_paths), idx)
        img_path = self.file_paths[idx]  # recall this is the nicely done reconstruction
        smap_path = random.choice(self.smaps)
        us_mask_path = random.choice(self.us_masks)

        # matter_path = self.matter_paths[idx]

        target_img = np.load(img_path)  # has size 1, 218, 170. 1 channel image, dims 218 x 170.
        # print ('target_img',target_img.shape)
        # if target_img.shape[-1] != 170:
        #     diff = int((target_img.shape[-1] - 170) / 2)  # difference per side
        #     target_img = target_img[:, :, diff:-diff]

        smap = np.load(smap_path)  # size channels, h, w

        # JUST FOR THE NEW CIRCLE SMAPS
        # print ('smaploaded', smap.shape)
        smap = smap[:, :, 0]
        smap = np.expand_dims(smap, axis=0)
        # print ('smapcut',smap.shape)
        # smap = np.moveaxis(smap, -1, 0)
        # print('smapmoved', smap.shape)
        mask = np.load(us_mask_path)
        # print ('maskloaded', mask.shape)
        mask = np.repeat(mask[None, :, :], self.coils, axis=0)
        # print ('maskrep', mask.shape)

        # Performing Contrast Inversion ..
        if self.invert != 'no':
            matter_path = self.matter_paths[idx]
            matter = np.load(matter_path)
            matter = (matter == 2) | (matter == 3)
            gray_mask = target_img * matter
            gray_mask = (gray_mask.max() - gray_mask) * matter
            target_img = target_img * (1 - matter) + gray_mask

        if target_img.shape[2] == 176:
            mask = mask[:, :, 40:-40]
            target_img = target_img[:, 0:-56, :]

        # input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap, axes=(-1, -2))), axes=(-1,
        # -2)) * mask
        input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img, axes=(-1, -2))), axes=(-1, -2)) * mask
        input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))),
                                    axes=(-1, -2))  # want shape channel h w

        input_img = np.moveaxis(input_img, 0,
                                -1)  # In numpy, want channels at end. Torch tensor transform will move them to the
        # front

        target_img = np.moveaxis(target_img, 0, -1)
        smap = np.moveaxis(smap, 0, -1)
        input_kspace = torch.from_numpy(input_kspace)
        mask = torch.from_numpy(mask)

        if self.data_transforms:  # realistically, ToTensor and then ReImgChannels
            input_img = self.data_transforms(input_img)
        if self.target_transforms:
            target_img = self.target_transforms(target_img)
            smap = self.target_transforms(smap)

        # scale by dividing all elements by the max value
        if input_img.dtype == torch.cdouble:
            input_max = torch.max(torch.view_as_real(input_img))
        else:
            input_max = torch.max(input_img)

        input_img = torch.div(input_img, input_max)
        target_img = torch.div(target_img, input_max)
        input_kspace = torch.div(input_kspace, input_max)
        input_max = torch.reshape(input_max, (1, 1, 1))

        return input_img, target_img, smap, input_max, input_kspace, mask
