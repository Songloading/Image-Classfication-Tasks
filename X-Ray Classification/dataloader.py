from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from glob import glob
import zipfile as z
from PIL import Image

def default_loader(path, transform, data_path):
    zf = z.ZipFile(data_path) 
    img_pil =  Image.open(zf.open(path))
    img_pil = img_pil.convert('RGB')
    img_tensor = transform(img_pil)
    return img_tensor

# Dataset Class
class X_Ray(Dataset):

    def __init__(self, data, target, transform, data_path, loader=default_loader ):
        self.data = data
        self.target = target
        self.transform = transform
        self.loader = loader
        self.data_path = data_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.target[idx]
        img = self.loader(path, self.transform, self.data_path)
        return [img, label]

def load_raw_data(data_path):
    # load csv file from zip
    zf = z.ZipFile(data_path) 
    df = pd.read_csv(zf.open('Data_Entry_2017.csv'))
    img_name = df.iloc[1, 0]
    df = df.loc[:, "Image Index":"Finding Labels"]

    # Data Preparation
    img_paths = {os.path.basename(name): name for name in zf.namelist() if name.endswith('.png')}
    df['path'] = df['Image Index'].map(img_paths.get)
    df.drop(['Image Index'], axis=1,inplace = True) # keep path and labels only

    # Make the data binary
    labels = df.loc[:,"Finding Labels"]
    one_hot = []
    for i in labels:
        if i == "No Finding":
            one_hot.append(0)
        else:
            one_hot.append(1)
    one_hot_series = pd.Series(one_hot)
    one_hot_series.value_counts()
    df['label'] = pd.Series(one_hot_series, index=df.index)
    df.drop(['Finding Labels'], axis=1,inplace = True)
    print(df.head())
    
    paths =  df['path'].tolist()
    targets = df['label'].tolist()
    train_paths = paths[:110000]
    train_targets = targets[:110000]
    test_paths =  paths[110000:]
    test_targets =  targets[110000:]
    return train_paths, train_targets, test_paths, test_targets



