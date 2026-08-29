import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import random
class Morph(torch.nn.Module):
    def __init__(self,in_ch, out_ch, kernel_size=3, stride = 1,padding=1,dilation=1, bias=False,operation="min"):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.operation = operation
        self.weights = torch.nn.Parameter(self.weight_initialise())
        self.bias = torch.nn.Parameter(torch.zeros(out_ch))

    def weight_initialise(self):
        w = torch.zeros(self.in_channels,self.kernel_size,self.kernel_size)
        init_choices = ["plus","diag","full","middle","col","row"]
        choice = random.choice(init_choices)
        mid_index = self.kernel_size//2

        if choice == "plus":
          w[:,mid_index,:] = 1
          w[:,:,mid_index] = 1

        if choice == "diag":
          for i in range(self.kernel_size):
            w[:,i,i] = 1

        if choice == "full":
          w[:,:,:] = 1

        if choice == "middle":
          w[:,mid_index,mid_index] = 1

        if choice == "col":
          w[:,mid_index,:] = 1

        if choice == "row":
          w[:,:,mid_index] = 1

        m = len(w[w > 0])
        w = w/m
        return w
    def calculate_out_dimensions(self,h_in, w_in):
        # Convert integers to tuples for height and width if they aren't already
        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        ph, pw = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        dh, dw = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)

        # Calculate H_out and W_out using floor division
        h_out = ((h_in + 2 * ph - dh * (kh - 1) - 1) // sh) + 1
        w_out = ((w_in + 2 * pw - dw * (kw - 1) - 1) // sw) + 1

        return h_out, w_out

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        N,C,H,W = x.shape # (N,C,H,W)
        patches = F.unfold(x,kernel_size=self.kernel_size,padding=self.padding,stride=self.stride,dilation=self.dilation) #(N,C*kw*kh,L)
        weights_flat = self.weights.reshape(1, -1, 1) # (1,C*kw*kh,1)
        # diff = torch.exp(patches - weights_flat) #(N,C*kw*kh,L)
        # output = -torch.log(torch.sum(diff,dim=1)) + self.bias #(N,L)
        # diff = patches - weights_flat
        if self.operation == "min":
            diff = patches - weights_flat
            output = torch.amin(diff,dim=1) + self.bias

        if self.operation == "max":
            diff = patches + weights_flat
            output = torch.amax(diff,dim=1) + self.bias

        output = output.unsqueeze(1) #(N,1,L)
        H_out,W_out = self.calculate_out_dimensions(H,W)
        folder = nn.Fold(output_size=(H_out, W_out), kernel_size=(1, 1), stride=1)
        return folder(output)

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
class Closing(nn.Module):
      def __init__(self,in_ch, out_ch, kernel_size=3, stride = 1,padding=1,dilation=1, bias=False):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.m_dilation = Morph(in_ch=in_ch,out_ch=out_ch,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="max")
        self.m_erosion = Morph(in_ch=in_ch,out_ch=out_ch,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="min")

      def forward(self,x):
        x = self.m_dilation(x)
        return self.m_erosion(x)

class Opening(nn.Module):
      def __init__(self,in_ch, out_ch, kernel_size=3, stride = 1,padding=1,dilation=1, bias=False):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.m_dilation = Morph(in_ch=in_ch,out_ch=out_ch,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="max")
        self.m_erosion = Morph(in_ch=in_ch,out_ch=out_ch,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="min")

      def forward(self,x):
        x = self.m_erosion(x)
        return self.m_dilation(x)

class Construction(nn.Module):
    def __init__(self,in_ch, out_ch, kernel_size=3, stride = 1,padding=1,dilation=1, bias=False):
      """
      In the constructor we instantiate four parameters and assign them as
      member parameters.
      """
      super().__init__()
      self.m_dilation = Morph(in_ch=in_ch,out_ch=out_ch,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="max")
      self.m_erosion = Morph(in_ch=in_ch,out_ch=out_ch,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="min")
      self.erosion_scale = nn.Parameter(torch.ones(1))
      self.dilation_scale = nn.Parameter(torch.ones(1))

    def forward(self,x):
      x = self.dilation_scale*self.m_dilation(x) + self.erosion_scale*self.m_erosion(x)
      return x

class ConcatMorph(nn.Module):
      def __init__(self,in_ch, kernel_size=3, stride = 1,padding=1,dilation=1, bias=False):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.m_dilation = Morph(in_ch=in_ch,out_ch=1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="max")
        self.m_erosion = Morph(in_ch=in_ch,out_ch=1,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,operation="min")

      def forward(self,x):
        return torch.cat([self.m_erosion(x),self.m_dilation(x)],dim=1)