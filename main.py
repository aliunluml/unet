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

if t.cuda.is_available():
    DEVICE = t.device("cuda")
else:
    DEVICE = t.device("cpu")

BATCH_SIZE=2
NUM_WORKERS=0
EPOCHS=20
LR=1e-3
MOMENTUM=0.9

def main():
    raw_dir=Path('raw')
    train_dataset=get_dataset(raw_dir/'train')
    test_dataset=get_dataset(raw_dir/'test')

    train_loader=t.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)
    test_loader=t.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=BATCH_SIZE,pin_memory=False,num_workers=NUM_WORKERS)

    diceloss=DiceLoss()
    net=UNet().to(DEVICE)
    opt=t.optim.SGD(net.parameters(),lr=LR,momentum=MOMENTUM)

    def train(model):
        model.train()
        losses=[]
        for subject_batch in train_loader:
            img_batch,seg_batch=preprocess(subject_batch)
            img_batch.to(DEVICE)
            seg_batch.to(DEVICE)

            logits=model(img_batch)

            loss=diceloss(logits,labels)
            losses.append(loss.item())
            loss.backward()

            opt.step()
            opt.zero_grad()

        return losses.mean()

    def test(model):
        model.eval()
        losses=[]
        with t.no_grad():
            for subject_batch in test_loader:
                img_batch,seg_batch=preprocess(subject_batch)
                img_batch.to(DEVICE)
                seg_batch.to(DEVICE)

                logits=model(img_batch)
                loss=diceloss(logits,labels)
                losses.append(loss.item())

        return losses.mean()

    dict={'epoch':[],'train_losses':[],'test_losses':[]}
    for epoch in range(0,EPOCHS):
        train_loss=train(net)
        test_loss=test(net)
        dict['train_losses'].append(train_loss)
        dict['test_losses'].append(test_loss)
        dict['epoch'].append(epoch)

    df=pd.DataFrame(dict)
    df.to_csv('out.csv')

    t.save(net.state_dict(),'weights.pt')

if __name__ == "__main__":
    main()
