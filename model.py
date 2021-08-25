import torch as t
import torch.nn.functional as F


class ContractingLayer(t.nn.Module):
    def __init__(self,depth,default=32):
        super(ContractingLayer,self).__init__()
        num_channel=2**depth*default
        self.max_pool=t.nn.MaxPool2d((2,2),stride=2)
        self.conv1=t.nn.Conv2d(num_channel,2*num_channel,(3,3),padding=1)
        self.conv2=t.nn.Conv2d(2*num_channel,2*num_channel,(3,3),padding=1)

    def forward(self,x):
        x=self.max_pool(x)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        return x


class ExpandingLayer(t.nn.Module):
    def __init__(self,depth,default=32):
        super(ExpandingLayer,self).__init__()
        num_channel=2**depth*default
        self.up_conv=t.nn.ConvTranspose2d(2*num_channel,num_channel,(2,2),stride=2)
        self.conv1=t.nn.Conv2d(2*num_channel,num_channel,(3,3),padding=1)
        self.conv2=t.nn.Conv2d(num_channel,num_channel,(3,3),padding=1)

    def forward(self,x,y):
        y=self.up_conv(y)
        out=t.cat((x,y),dim=1)
        out=F.relu(self.conv1(out))
        out=F.relu(self.conv2(out))
        return out

class UNet(t.nn.Module):
    def __init__(self,num_layers=4,default=32):
        super(UNet, self).__init__()
        self.num_layers=num_layers

        self.conv1=t.nn.Conv2d(1,default,(3,3),padding=1)
        self.conv2=t.nn.Conv2d(default,default,(3,3),padding=1)

        for i in range(0,num_layers):
            setattr(self, 'expandinglayer'+str(i), ExpandingLayer(i,default))
            setattr(self, 'contractinglayer'+str(i), ContractingLayer(i,default))

        self.final_conv=t.nn.Conv2d(default,1,(1,1))

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        skips=[x]

        # Going inward (Contraction) with increasing order. Index is depth
        indices=list(range(0,self.num_layers))
        for i in indices:
            contractinglayer=getattr(self,'contractinglayer'+str(i))
            y=contractinglayer(skips[i])
            skips.append(y)

        out=skips.pop()

        # Going outward (Expansion) with reverse order. Index is depth
        indices.reverse()
        for i in indices:
            expandinglayer=getattr(self,'expandinglayer'+str(i))
            out=expandinglayer(skips[i],out)

        out=F.sigmoid(self.final_conv(out))

        return out
