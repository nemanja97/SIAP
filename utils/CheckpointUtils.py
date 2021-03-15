import os

import torch

from utils.Constants import MODELS_SAVE_LOCATION


def save_checkpoint(checkpoint, file_name):
    torch.save(checkpoint, os.path.join(MODELS_SAVE_LOCATION, file_name))


def load_checkpoint(file_name, device):
    return torch.load(os.path.join(MODELS_SAVE_LOCATION, file_name), map_location=torch.device(device))
