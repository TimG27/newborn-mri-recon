"""
This file defines functions that are used to convert a 1-coil image to a 12-coil image
"""


import csv
import os
import sys

# Required for bart to work
path = os.environ["TOOLBOX_PATH"] + "/python/"
sys.path.append(path)

import bart
import numpy as np
import h5py
import SimpleITK as sitk


def pad_3d_array(arr, desired_shape):
    """
    Pads an array to the desired shape
    Args:
        arr: Array to be padded
        desired_shape: Shape of the desired size
    Returns: Padded array
    """
    current_shape = arr.shape
    if current_shape == desired_shape:
        return arr

    padded_arr = np.zeros(desired_shape, dtype=arr.dtype)

    pad_x = max(0, desired_shape[0] - current_shape[0])
    pad_y = max(0, desired_shape[1] - current_shape[1])
    pad_z = max(0, desired_shape[2] - current_shape[2])

    start_x, start_y, start_z = pad_x // 2, pad_y // 2, pad_z // 2

    end_x = start_x + current_shape[0]
    end_y = start_y + current_shape[1]
    end_z = start_z + current_shape[2]

    padded_arr[start_x:end_x, start_y:end_y, start_z:end_z] = arr

    return padded_arr


def pre_reg_MRI(MRI_image):
    """
    This function applies empirical alignment for MRI image
    :param MRI_image: The image array
    :return: Adjusted image
    """
    return MRI_image[:, :, 40:256]


def pre_reg_Ref(Ref_image):
    """
    This function applies empirical alignment for reference image
    :param Ref_image: The reference image array
    :return: Adjusted image
    """
    ref = np.flip(Ref_image, axis=2)
    ref = np.flip(np.rot90(ref, k=3), axis=0)
    ref = ref[:, :, 0:216]
    ref = pad_3d_array(ref, (200, 256, 216))
    return ref


def gen_espirit_coils(file_name, MRI_path, kspace_path, transform_path, acq_path):
    """
    Given the image, generate the coil sensitivity maps using espirit in the bart toolbox
    :param file_name: File name of the csv
    :param MRI_path: Path with the original MRI images
    :param kspace_path: Path where all the k-space images are stored
    :param transform_path: The transform that needs to be applied.
    :param acq_path: Path to save the acquisitions
    :return:
    """

    csv_file = file_name

    MRI_files = []
    Ref_files = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            MRI_files.append(row[0])
            Ref_files.append(row[1])

    for MRI, Ref in zip(MRI_files, Ref_files):
        MRI_image = np.load(MRI_path + MRI + '.npy')

        Ref_image = str(kspace_path + Ref[0:15] + '.7.h5')

        kspace = h5py.File(Ref_image)['kspace'][:]

        """
        plt.imshow(kspace[100, :, :, 0], cmap='gray')
        plt.show()
        """

        # Convert to complex
        kspace = kspace[:, :, :, ::2] + 1j * kspace[:, :, :, 1::2]

        print(kspace.shape)

        map_vol = []

        # Doing all the slice
        for i in range(0, 256):
            slice = i
            k_slice = kspace[slice, :, :, :]
            k_slice = k_slice[None, :, :, :]

            kspace_fftmod = bart.bart(1, 'fftmod 6', k_slice)
            smap = np.squeeze(bart.bart(1, 'ecalib -m 1', kspace_fftmod))

            map_vol.append(smap)

        map_vol = np.stack(map_vol)
        print(map_vol.shape)

        acquisition = []

        for i in range(0, 12):
            map1 = np.transpose(map_vol[:, :, :, i], (1, 2, 0))

            # Preregistration alignment
            map1 = pre_reg_Ref(map1)
            fixed_image = pre_reg_MRI(MRI_image)

            transform = sitk.ReadTransform(str(transform_path + MRI + '.tfm'))
            resampler = sitk.ResampleImageFilter()

            fixed_image = sitk.GetImageFromArray(fixed_image)

            map1 = np.abs(map1)
            map1 = sitk.GetImageFromArray(map1)
            map1 = sitk.Cast(map1, sitk.sitkFloat32)

            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(transform)

            print(sitk.GetArrayFromImage(map1).shape, sitk.GetArrayFromImage(fixed_image).shape)

            trans_map = resampler.Execute(map1)

            print(sitk.GetArrayFromImage(trans_map).shape)

            # MAKE THE COIL ACQUISITION

            coil_acq = np.multiply(sitk.GetArrayFromImage(trans_map), sitk.GetArrayFromImage(fixed_image))
            print('Dot product done')

            acquisition.append(coil_acq)

        acquisition = np.stack(acquisition)
        print('Acqusition shape', acquisition.shape)

        np.save(str(acq_path + MRI), acquisition)
