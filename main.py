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
BATCH_SIZE=16
NUM_WORKERS=2
EPOCHS=20


def get_dataset(dataset_dir):
    seg_paths = sorted(dataset_dir.glob('*liver.nii.gz'))
    img_paths = sorted(dataset_dir.glob('*orig.nii.gz'))

    subjects=[]
    for seg_path,img_path in zip(seg_paths,img_paths):
        subject = tio.Subject(mri=tio.ScalarImage(img_path),liver=tio.LabelMap(seg_path))
        subjects.append(subject)

    # Preprocessing
    hist_landmarks = tio.HistogramStandardization.train(img_paths,output_path=dataset_dir/'landmarks.npy')
    hist_transform = tio.HistogramStandardization({'mri': hist_landmarks})
    znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    transforms=tio.Compose([hist_transform,znorm_transform])

    dataset = tio.SubjectsDataset(subjects,transform=transforms)
    return dataset


raw_dir='raw'
train_dataset=get_dataset(raw_dir/'train')
test_dataset=get_dataset(raw_dir/'test')

train_loader=t.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)
test_loader=t.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)

model=

for epoch in range(0,EPOCHS):
