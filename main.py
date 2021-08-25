from pathlib import Path
import numpy as np
import torch as t
import torchvision as tv
import torchio as tio
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from models import UNet
from lib import DiceLoss,get_dataset,preprocess

t.backends.cudnn.benchmark=True
BATCH_SIZE=5
NUM_WORKERS=0
EPOCHS=20

# def train(net):
#     loss=0.
#     for


raw_dir=Path('raw')
train_dataset=get_dataset(raw_dir/'train')
test_dataset=get_dataset(raw_dir/'test')

train_loader=t.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)
test_loader=t.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)

diceloss=DiceLoss()

# def train(model):
#     for subject_batch in train_loader:
#         img_batch,seg_batch=preprocess(subject_batch)
#         logits=model(img_batch)
#         loss=diceloss(logits,labels)
#         loss.backward()
#         break



# model=UNet()
# opt=
#
# for epoch in range(0,EPOCHS):
