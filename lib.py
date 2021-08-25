import torch as t


class DiceLoss(t.nn.Module):
    def __init__(self,smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth=smooth

    def forward(self,logits,labels):
        intersection=2*(logits*labels).sum()
        union=logits.sum()+labels.sum()
        out= (intersection+self.smooth)/(union+self.smooth)
        return -out
