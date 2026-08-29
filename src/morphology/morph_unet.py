from segmentation_models_pytorch import Unet
from .morphology import Morph, ConcatMorph
import torch
import torch.nn as nn

class MorphUnet(nn.Module):
      def __init__(self,in_ch,kernel_size=3, stride = 1,padding=1,dilation=1, bias=False):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.erosion = nn.ModuleList(
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation),
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation),
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation),
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        )
        self.dilation = nn.Sequential(
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,operation="max"),
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,operation="max"),
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,operation="max"),
            Morph(in_ch,1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,operation="max")
        )
        self.morph = ConcatMorph(in_ch,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.unet = Unet(encoder_name="resnet50",encoder_weights="imagenet",in_channels=in_ch)

      def forward(self,x):
        x = torch.mean(torch.cat([op[x] for op in self.erosion],dim=1),dim=1)
        x = self.unet(x)
        x = torch.mean(torch.cat([op[x] for op in self.dilation],dim=1),dim=1)
        return x