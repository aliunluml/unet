from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch as t
import torchvision as tv
import torchio as tio
import matplotlib.pyplot as plt

from models import UNet
from lib import DiceLoss,get_dataset,preprocess

t.backends.cudnn.benchmark=True

if t.cuda.is_available():
    DEVICE = t.device("cuda")
else:
    DEVICE = t.device("cpu")


def get_model(args):
    raw_dir=Path('raw')
    train_dataset=get_dataset(raw_dir/'train')
    test_dataset=get_dataset(raw_dir/'test')

    train_loader=t.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,pin_memory=False,num_workers=0)
    test_loader=t.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=args.batch_size,pin_memory=False,num_workers=0)

    diceloss=DiceLoss()
    net=UNet().to(DEVICE)
    opt=t.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)

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
    for epoch in range(0,args.epochs):
        train_loss=train(net)
        test_loss=test(net)
        dict['train_losses'].append(train_loss)
        dict['test_losses'].append(test_loss)
        dict['epoch'].append(epoch)

    df=pd.DataFrame(dict)
    df.to_csv('out.csv')

    t.save(net.state_dict(),'weights.pt')
    return net

def run(model):
    raw_dir=Path('raw')
    test_dataset=get_dataset(raw_dir/'test')
    test_loader=t.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=1,pin_memory=False,num_workers=0)
    batch=next(iter(test_loader))
    mri,liver=preprocess(batch)
    with t.no_grad():
        mri.to(DEVICE)
        probs=model(mri).cpu()
        mri.cpu()
    affine = batch['mri'][tio.AFFINE][0].numpy()
    subject = tio.Subject(mri=tio.ScalarImage(tensor=t.movedim(mri,0,-1), affine=affine),liver=tio.LabelMap(tensor=t.movedim(liver,0,-1), affine=affine),predicted=tio.ScalarImage(tensor=t.movedim(probs,0,-1), affine=affine))
    # subject.plot(figsize=(9, 8), cmap_dict={'predicted': 'RdBu_r'})
    subject['mri'].save('orig.nii.gz')
    subject['liver'].save('liver.nii.gz')
    subject['predicted'].save('pred.nii.gz')


def main(args):
    if args.filename is not None:
        model=UNet()
        model.load_state_dict(t.load(args.filename))
        model.to(DEVICE)
        model.eval()
    else:
        model=get_model(args)
        model.eval()

    run(model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',   type=str,   nargs=1)
    parser.add_argument('--epochs',     type=int,   nargs=1, default=20)
    parser.add_argument('--lr',         type=float, nargs=1, default=1e-3)
    parser.add_argument('--batch_size', type=int,   nargs=1, default=2)
    args=parser.parse_args()
    main(args)
