"""
This file runs a validation loop on the newborn dataset
"""

import glob
import os
import random
from datetime import datetime

import pandas as pd
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from utils.cascaded_model_architecture import CascadedModel
from utils.dataset_generator import SliceSmapDataset, ReImgChannels
from utils.metrics import SSIM as SSIM_numpy
from utils.training_utils import *

dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

print('imports complete')

# WANDB PARAMETERS #
project = 'inverted cascaded model'
run_name = 'inversion corr-try newborn bigslice'

smap = "circle_R_8"
epochs = 1
coils = 12
version = 2
blocks = 5
block_depth = 5
filters = 110
smap_layers = 12
smap_filters = 110
batch_size = 4
lr = 0.001
stopper_patience = 10
plateau_patience = 6
clip = 50
optim_name = 'Adam'
weight_decay = 0.1
note = ""

id = wandb.util.generate_id()
config = {
    "model_type": "smap_branch_cascaded",
    "version": version,
    "coils": coils,
    "epochs": epochs,
    "blocks": blocks,
    "block_depth": block_depth,
    "filters": filters,
    "smap_layers": smap_layers,
    "smap_filters": smap_filters,
    "batch_size": batch_size,
    "learning_rate": lr,
    #     "loss_function": los/s_fn,
    "optimizer": optim_name,
    "weight_decay": weight_decay,
    "gradient_clipping": clip,
    "reduce_lr_patience": plateau_patience,
    "early_stopper_patience": stopper_patience,
    "date/time": dt_string,
    "run_id": id,
}
run = wandb.init(project=project, id=id, name=run_name, config=config, notes=note)

# initiate some random seeds and check cuda
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

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

valid_data = SliceSmapDataset(slice_ids, 'valid', smaps, masks, 'espirit', coils, data_transforms=test_transforms,
                              target_transforms=test_transforms, invert='no')

# create dataloaders
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, drop_last=True)

# define model
input_channels = coils * 2
model = CascadedModel(input_channels, reps=blocks, block_depth=block_depth, filters=filters,
                      smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)

checkpoint = torch.load(os.path.join('model_weights', f'cinverted_cascaded_model_v{version}.pt'))
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

# define hyperparameters
criterion_mse = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=plateau_patience)
early_stopper = EarlyStopper(patience=stopper_patience)

i = 0
best_loss = 1e20

val_loss = 0
### VALIDATION LOOP ###
with torch.no_grad():
    preds = []
    labels = []
    smaps = []
for i, data in enumerate(valid_loader, 0):
    input, img_label, smap_label = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.float32), \
        data[2].to(device, dtype=torch.float32)
    scale_value, input_kspace, mask = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), data[
        5].to(device, dtype=torch.float32)

    output_img, output_smap = model((input, input_kspace, mask))

    loss = criterion_mse(output_img, to_rms(img_label))

    val_loss += loss.item()

    if i in range(40):
        mask = mask.detach().cpu().numpy()
        R = 1 / (np.sum(mask) / np.size(mask))
        img_pred = output_img.detach().cpu().numpy()
        img_label = to_rms(img_label.detach().cpu()).numpy()
        img_pred, img_label = np.abs(img_pred[0, 0, :, :] + 1j * img_pred[0, 1, :, :]), np.abs(
            img_label[0, 0, :, :] + 1j * img_label[0, 1, :, :])
        ssim = SSIM_numpy(img_pred, img_label)
        caption = f"R={R:.1f}, SSIM: {ssim:.3f}"
        smap_pred = output_smap.detach().cpu().numpy()
        smap_pred = np.abs(smap_pred[0, 0, :, :] + 1j * smap_pred[0, 1, :, :])

        preds.append(wandb.Image(wandb_scale_img(img_pred), caption=caption))
        labels.append(wandb.Image(wandb_scale_img(img_label)))
        smaps.append(wandb.Image(wandb_scale_img(smap_pred)))

    val_loss /= (i + 1)
    print(f'val loss: {val_loss:.6f}', flush=True)

    wandb.log({
        # "train_loss": train_loss,
        "val_loss": val_loss,
        'pred': preds,
        'pred_label': labels,
        'smap': smaps},
    )
