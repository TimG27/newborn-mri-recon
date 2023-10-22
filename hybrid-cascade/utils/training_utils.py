"""
This file defines some utility functions to help in training.
"""

import numpy as np
import torch


class EarlyStopper:
    """ Stop training if validation loss does not seem to decrease """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_checkpoint(checkpoint_fpath, model, optimizer):
    """
    Loads a model from a previously saved checkpoint
    Args:
        checkpoint_fpath (string): Path to the stored mode
        model: The newly defined model
        optimizer: The training optimizer used
    Returns: The saved model
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['best_loss']


def wandb_scale_img(data: np.ndarray) -> np.ndarray:
    """ Scales float data to a range of 0, 1
    """
    d_max = np.max(data)
    d_min = np.min(data)
    data = (data - d_min) / (d_max - d_min)
    return data


def to_rms(tensor):
    """ Given a 12 channel image of real and imaginary parts, combine them into a single image by calculating the
    root-mean-square"""

    even_channels = tensor[:, ::2, :, :]
    odd_channels = tensor[:, 1::2, :, :]

    rms_even = torch.sqrt(torch.mean(even_channels ** 2, dim=1, keepdim=True))
    rms_odd = torch.sqrt(torch.mean(odd_channels ** 2, dim=1, keepdim=True))

    rms_array = torch.cat([rms_even, rms_odd], dim=1)
    return rms_array
