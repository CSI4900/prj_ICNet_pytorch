"""Model store which provides pretrained models."""
from __future__ import print_function

import os

from utils.download import download

__all__ = ['get_model_file', 'get_resnet_file']

model_urls = {
    'resnet50-25c4b509': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth',
    'resnet101-2a57e44d': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth',
    'resnet152-0d43d698': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth',
}


def get_resnet_file(name, root=None):
    if name not in model_urls:
        raise ValueError(f'Pretrained model for {name} is not available.')

    # Set model save path
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    file_path = os.path.join(root, f"{name}.pth")

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Using cached model file: {file_path}")
        return file_path

    # Download model using utils.download
    print('Model file {} is not found. Downloading.'.format(file_path))
    url = model_urls[name]
    print(f"Downloading {name} from {url} to {file_path}...")
    download(url, path=file_path, overwrite=False)  # Keep existing files

    print(f"Download complete: {file_path}")
    return file_path


def get_model_file(name, root='~/.torch/models'):
    root = os.path.expanduser(root)
    file_path = os.path.join(root, name + '.pth')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Downloading or trainning.')
