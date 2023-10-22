"""
This file contains the training loop that trains the cascade model on the image slices.
"""

import glob
import random
from datetime import datetime

# import numpy as np
import pandas as pd
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

# from loss_functions import SSIMLoss
from utils.cascaded_model_architecture import CascadedModel
from utils.dataset_generator import SliceSmapDataset, ReImgChannels
from utils.metrics import SSIM as SSIM_numpy
from utils.training_utils import *

dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# print('imports complete')

project = 'cascaded model'
run_name = 'updated dataloader'

smap = "circle_R_8"
epochs = 100
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
smap_style = ''
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
    #  "loss_function": los/s_fn,
    "optimizer": optim_name,
    "weight_decay": weight_decay,
    "gradient_clipping": clip,
    "reduce_lr_patience": plateau_patience,
    "early_stopper_patience": stopper_patience,
    "date/time": dt_string,
    "run_id": id,
    "smap_style": smap_style,
}
# run_name = f"smap-cascaded=model-till-65"
run = wandb.init(project=project, id=id, name=run_name, config=config, notes=note)  # resume is True when resuming

# initiate some random seeds and check cuda
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

folder_path = '/home/timothy2/scratch/cascade-try/'

slice_ids = pd.read_csv(folder_path + 'slice_ids_v5.csv')
# slice_ids = pd.read_csv ('/home/timothy/PycharmProjects/undsamp/folder-slice-view/smaller_version/slice_ids2.csv')
test_transforms = transforms.Compose(
    [
        ToTensor(),
        ReImgChannels()
    ]
)

# generate datasets
smaps = glob.glob('/home/timothy2/projects/def-rmsouza/timothy2/model-translation/coil-tiles-try/coil_data/*.npy')
masks = glob.glob(r'/home/timothy2/projects/def-rmsouza/timothy2/model-translation/undersamp/masks-100-2/*.npy')

train_data = SliceSmapDataset(slice_ids, 'train', smaps, masks, 'espirit', coils, data_transforms=test_transforms,
                              target_transforms=test_transforms)
valid_data = SliceSmapDataset(slice_ids, 'valid', smaps, masks, 'espirit', coils, data_transforms=test_transforms,
                              target_transforms=test_transforms)

# create dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, drop_last=True)

print('train-', len(train_loader), 'valid-', len(valid_loader))

# define model
input_channels = coils * 2
model = CascadedModel(input_channels, reps=blocks, block_depth=block_depth, filters=filters,
                      smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)
model.to(device)
model_save_path = f'model_weights/smap_branch_cascaded_model_v{version}.pt'

# define hyperparameters
criterion_mse = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=plateau_patience)
early_stopper = EarlyStopper(patience=stopper_patience)

# model, optimizer, start_epoch, loss = load_checkpoint(model_save_path, model, optimizer)

# print('train-', len(train_loader), 'valid-', len(valid_loader))
best_loss = 1e20

### TRAIN LOOP ###
print(f'Started training model version {version}', flush=True)
for epoch in range(epochs):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, img_labels, smap_labels = data[0].to(device, dtype=torch.float32), data[1].to(device,
                                                                                              dtype=torch.float32), \
            data[2].to(device, dtype=torch.float32)
        scale_values, input_kspaces, masks = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), \
            data[5].to(device, dtype=torch.float32)

        optimizer.zero_grad()
        output_imgs, output_smaps = model((inputs, input_kspaces, masks))

        loss = criterion_mse(output_imgs, to_rms(img_labels)) + criterion_mse(to_rms(output_smaps),
                                                                              smap_labels)  # + loss_ssim
        # loss = criterion_l1(output_imgs, img_labels) + criterion_l1(output_smaps, smap_labels) + loss_ssim
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    train_loss /= (i + 1)
    print(f'{epoch + 1},  train loss: {train_loss:.6f}', flush=True)

    val_loss = 0
    ### VALIDATION LOOP ###
    with torch.no_grad():
        preds = []
        labels = []
        smaps = []
        for i, data in enumerate(valid_loader, 0):
            input, img_label, smap_label = data[0].to(device, dtype=torch.float32), data[1].to(device,
                                                                                               dtype=torch.float32), \
                data[2].to(device, dtype=torch.float32)
            scale_value, input_kspace, mask = data[3].to(dtype=torch.float32), data[4].to(device,
                                                                                          dtype=torch.complex64), data[
                5].to(device, dtype=torch.float32)

            output_img, output_smap = model((input, input_kspace, mask))

            loss = criterion_mse(output_img, to_rms(img_label)) + criterion_mse(to_rms(output_smap),
                                                                                smap_label)  # + loss_ssim
            # loss = criterion_l1(output_img, img_label) + criterion_l1(output_smap, smap_label) + loss_ssim
            val_loss += loss.item()

            if i in range(4):
                mask = mask.detach().cpu().numpy()
                R = 1 / (np.sum(mask) / np.size(mask))
                img_pred = output_img.detach().cpu().numpy()
                img_label = to_rms(img_label.detach().cpu())
                img_label = img_label.numpy()
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
        scheduler.step(val_loss)

        wandb.log({"train_loss": train_loss,
                   "val_loss": val_loss,
                   'pred': preds,
                   'pred_label': labels,
                   'smap': smaps}, step=epoch + 1)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving model", flush=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, model_save_path
            )

    if early_stopper.early_stop(val_loss):
        nepochs = epoch + 1
        break

print('Finished Training')
