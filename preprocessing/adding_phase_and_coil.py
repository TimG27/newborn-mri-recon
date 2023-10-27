"""
We execute this file to complete phase and coil configuration to our single channel, magnitude only images.
"""

from preproccessing.utils.phase_utils import *

# PHASE ADDITION

print('Beginning phase testing...')

folder1_path = '/home/timothy2/projects/def-rmsouza/timothy2/mini_mri_dataset/adult_60_numpy/'
folder2_path = '/home/timothy2/projects/def-rmsouza/timothy2/mini_mri_dataset/phase_adults_19/'
destination_path = '/home/timothy2/projects/def-rmsouza/timothy2/mini_mri_dataset/complex_adult_60_v2/'
trans_path = '/home/timothy2/projects/def-rmsouza/timothy2/model-translation/coil-tiles-v2/saved_transforms/'

random_add_phase(folder1_path, folder2_path, destination_path, trans_path)

print('Phase testing done..')

print('Beginning coil testing..')

# ADDING 12 COIL CONFIGURATION

csv = 'file_mapping.csv'
MRI_path = '/home/timothy2/projects/def-rmsouza/timothy2/mini_mri_dataset/complex_adult_60_v2/'
kspace_path = '/home/timothy2/projects/def-rmsouza/timothy2/mini_mri_dataset/kspace_adults_19/'
transform_path = '/home/timothy2/projects/def-rmsouza/timothy2/model-translation/coil-tiles-v2/saved_transforms/'
acq_path = '/home/timothy2/projects/def-rmsouza/timothy2/model-translation/coil-tiles-v2/coiled_adult_60/'

gen_espirit_coils(csv, MRI_path, kspace_path, transform_path, acq_path)

print('Coil testing done..')
