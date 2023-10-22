"""
This file runs a validation loop on the newborn dataset
"""

import glob
import random
from datetime import datetime
import os

# import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

import wandb
# from cascaded_map_branch import CascadedModel, to_complex
from unet.utils.dataset_generator import SliceSmapDataset, ReImgChannels
# from loss_functions import SSIMLoss
from unet.utils.metrics import SSIM as SSIM_numpy
from unet.utils.training_utils import *
from unet.utils.unet_architecture import UNet

print('imports complete')

dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

project = 'unet model'
run_name = 'newborns-bigslice'

smap = "circle_R_8"
epochs = 100
coils = 12
version = 2
blocks = 4
block_depth = 4
filters = 2048
# smap_layers = 12
# smap_filters = 110
batch_size = 4
lr = 0.001
stopper_patience = 10
plateau_patience = 6
# clip = 50
optim_name = 'Adam'
weight_decay = 0.1
# smap_style = ''
note = ""

id = wandb.util.generate_id()
config = {
    "model_type": "unet model",
    "version": version,
    "coils": coils,
    "epochs": epochs,
    "blocks": blocks,
    "block_depth": block_depth,
    "filters": filters,
    # "smap_layers": smap_layers,
    # "smap_filters": smap_filters,
    "batch_size": batch_size,
    "learning_rate": lr,
    # "loss_function": loss_fn,
    "optimizer": optim_name,
    "weight_decay": weight_decay,
    # "gradient_clipping": clip,
    "reduce_lr_patience": plateau_patience,
    "early_stopper_patience": stopper_patience,
    "date/time": dt_string,
    "run_id": id,
    # "smap_style": smap_style,
}
# run_name = f"unet-model-till-35"
run = wandb.init(project=project, id=id, name=run_name, config=config, notes=note)  # resume is True when resuming

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

folder_path = '/home/timothy2/scratch/cascade-try/'

slice_ids = pd.read_csv(folder_path + 'slice_ids_v5.csv')
slice_ids = pd.read_csv('/home/timothy2/scratch/newborn-try/slice_ids-test.csv')

test_transforms = transforms.Compose(
    [
        ToTensor(),
        ReImgChannels()
    ]
)

# generate datasets
smaps = glob.glob('/home/timothy2/projects/def-rmsouza/timothy2/model-translation/coil-tiles-try/coil_data/*.npy')
masks = glob.glob(r'/home/timothy2/projects/def-rmsouza/timothy2/model-translation/undersamp/masks-100-3/*.npy')

valid_data = SliceSmapDataset(slice_ids, 'valid', smaps, masks, 'espirit', coils=12, data_transforms=test_transforms,
                              target_transforms=test_transforms)

# create dataloaders
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, drop_last=True)

# print ('valid - ', len(valid_loader))

input_channels = coils * 2
model = UNet(in_channels=input_channels, out_channels=24).type(torch.float32)

checkpoint = torch.load(os.path.join('model_weights', f'unet_model_v{version}.pt'))
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

criterion_mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6)
early_stopper = EarlyStopper(patience=stopper_patience)
best_loss = 1e20

val_loss = 0
### VALIDATION LOOP ###
with torch.no_grad():
    preds = []
    labels = []
    smaps = []
for i, data in enumerate(valid_loader, 0):

    input_img, img_labels = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.float32)
    # print (input_img.shape, img_labels.shape)

    output_img = model(input_img)

    loss = criterion_mse(output_img, img_labels)

    val_loss += loss.item()

    if i in range(40):
        img_pred = to_rms(output_img.detach().cpu())
        img_label = to_rms(img_labels.detach().cpu())
        img_pred, img_label = np.abs(img_pred[0, 0, :, :] + 1j * img_pred[0, 1, :, :]), np.abs(
            img_label[0, 0, :, :] + 1j * img_label[0, 1, :, :])
        img_pred, img_label = img_pred.numpy(), img_label.numpy()
        ssim = SSIM_numpy(img_pred, img_label)
        caption = f"SSIM: {ssim:.3f}"

        preds.append(wandb.Image(wandb_scale_img(img_pred), caption=caption))
        labels.append(wandb.Image(wandb_scale_img(img_label)))

    val_loss /= (i + 1)
    print(f'val loss: {val_loss:.6f}', flush=True)

    wandb.log({
        "val_loss": val_loss,
        'pred': preds,
        'pred_label': labels,
    },
    )
