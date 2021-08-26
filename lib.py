import torch as t
import torchio as tio
import torchvision as tv



IMG_HEIGHT=512
IMG_WIDTH=512
IMG_DEPTH=128

def get_transforms(landmarks):
    ras_transform=tio.ToCanonical()
    crop_transform=tio.CropOrPad((IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH))
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

def slice3D(x):
    # (Batch,Channel.Height,Width,Depth) -> (Batch*Depth,Channel,Height,Width)
    x=t.movedim(x,-1,0)
    x=t.flatten(x,start_dim=0,end_dim=1)
    return x

def preprocess(subject_batch):
    imgs=slice3D(subject_batch['mri']['data'])
    segs=slice3D(subject_batch['liver']['data'])
    resize=tv.transforms.Resize((IMG_HEIGHT//2,IMG_WIDTH//2))
    imgs=resize(imgs)
    segs=resize(segs)
    return imgs,segs



class DiceLoss(t.nn.Module):
    def __init__(self,smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth=smooth

    def forward(self,logits,labels):
        intersection=2*(logits*labels).sum()
        union=logits.sum()+labels.sum()
        out= (intersection+self.smooth)/(union+self.smooth)
        return -out
