from pathlib import Path
import numpy as np
import torch as t
import torchvision as tv
import torchio as tio
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from models import UNet
from lib import DiceLoss

t.backends.cudnn.benchmark=True
BATCH_SIZE=5
NUM_WORKERS=0
EPOCHS=20

def get_transforms(landmarks):
    ras_transform=tio.ToCanonical()
    crop_transform=tio.CropOrPad((512,512,128))
    hist_transform = tio.HistogramStandardization(landmarks)
    znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)

    transforms=tio.Compose([ras_transform,crop_transform,hist_transform,znorm_transform])
    return transforms

def get_dataset(dataset_dir):
    seg_paths = sorted(dataset_dir.glob('*liver.nii.gz'))
    img_paths = sorted(dataset_dir.glob('*orig.nii.gz'))

    subjects=[]
    for seg_path,img_path in zip(seg_paths,img_paths):
        subject = tio.Subject(mri=tio.ScalarImage(img_path),liver=tio.LabelMap(seg_path))
        subjects.append(subject)

    # Preprocessing
    hist_landmarks = tio.HistogramStandardization.train(img_paths,output_path=dataset_dir/'landmarks.npy')
    dict={'mri': hist_landmarks}
    transforms=get_transforms(dict)
    dataset = tio.SubjectsDataset(subjects,transform=transforms)
    return dataset

# def train(net):
#     loss=0.
#     for


raw_dir=Path('raw')
train_dataset=get_dataset(raw_dir/'train')
test_dataset=get_dataset(raw_dir/'test')

train_loader=t.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)
test_loader=t.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)

for subjects in train_loader:
    print(subjects['mri']['data'].shape)
    print(subjects['liver']['data'].shape)
    break



# model=UNet()
# opt=
#
# for epoch in range(0,EPOCHS):
