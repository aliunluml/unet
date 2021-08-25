import torch as t


class DiceLoss(t.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self):
