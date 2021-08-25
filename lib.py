import torch as t
import torchio as tio
import torchvision as tv



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


# class LiverDataset(t.utils.data.Dataset):
#     def __init__(self,path,undersample=True,factor=2):
#         super(LiverDataset,self).__init__()
#         self.sdataset=get_dataset(path)
#         self.factor=factor if undersample else 1
#
#     def __len__(self):
#         return len(self.sdataset)
#
#     def slice(self,x):
#         print(x.shape)
#         x=t.movedim(x,-1,0)
#         print(x.shape)
#         x=t.flatten(x,start_dim=0,end_dim=1)
#         print(x.shape)
#         return x
#
#     def __getitem__(self,i):
#         sbatch=self.sdataset.__getitem__(i)
#         imgs=self.slice(sbatch['mri']['data'])
#         segs=self.slice(sbatch['liver']['data'])
#
#         imgs=imgs[:,::self.factor,::self.factor]
#         segs=segs[:,::self.factor,::self.factor]
#
#         return imgs,segs
#
# def slice()

# def preprocess(subjects):
#     imgs=self.slice(subjects['mri']['data'])
#     segs=self.slice(subjects['liver']['data'])



class DiceLoss(t.nn.Module):
    def __init__(self,smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth=smooth

    def forward(self,logits,labels):
        intersection=2*(logits*labels).sum()
        union=logits.sum()+labels.sum()
        out= (intersection+self.smooth)/(union+self.smooth)
        return -out
