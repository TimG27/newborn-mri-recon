"""
This file test runs the model on a random under-sampled image
"""

import glob
import random

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from utils.cascaded_model_architecture import *
from utils.dataset_generator import SliceSmapDataset, ReImgChannels
from utils.loss_functions import *
from utils.training_utils import *

blocks = 5
block_depth = 5
filters = 110
smap_layers = 12
smap_filters = 110

# print('imports complete')

# initiate some random seeds and check cuda
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

folder_path = '/home/timothy2/scratch/cascade-try/'

slice_ids = pd.read_csv(folder_path + 'slice_ids_v3.csv')
test_transforms = transforms.Compose(
    [
        ToTensor(),
        ReImgChannels()
    ]
)

# generate datasets
smaps = glob.glob('/home/timothy2/projects/def-rmsouza/timothy2/model-translation/coil-tiles-try/coil_data/*.npy')
masks = glob.glob(r'/home/timothy2/projects/def-rmsouza/timothy2/model-translation/undersamp/masks-100-2/*.npy')

test_data = SliceSmapDataset(slice_ids, 'test', smaps, masks, 'espirit', coils=8, data_transforms=test_transforms,
                             target_transforms=test_transforms)

# create dataloader
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True)

# print('test = ', len(test_loader))

# define model
input_channels = 8 * 2
model = CascadedModel(input_channels, reps=blocks, block_depth=block_depth, filters=filters,
                      smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)

model.to(device)
model_save_path = f'model_weights/smap_branch_cascaded_model_v0.pt'

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)

model, optimizer, start_epoch, loss = load_checkpoint(model_save_path, model, optimizer)

# define hyperparameters
criterion_ssim = SSIMLoss()

for i in range(0, 2000):
    data = next(iter(test_loader))

inputs, img_labels, smap_labels = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.float32), \
    data[2].to(device, dtype=torch.float32)
scale_values, input_kspaces, masks = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), data[
    5].to(device, dtype=torch.float32)

input_img = inputs

# print(input_img.shape, img_labels.shape)

output_imgs, output_smaps = model((inputs, input_kspaces, masks))

# print(output_imgs.shape, img_labels.shape)

np.save('random-pred.npy', output_imgs.cpu().detach().numpy())
np.save('random-truth.npy', img_labels.cpu().detach().numpy())
