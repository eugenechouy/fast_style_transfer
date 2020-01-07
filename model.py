import torch
from  torchvision import models
from collections import namedtuple

class ImageTransformNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # convolution layers
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.norm1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.norm2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.norm3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # deconvolution layers 
        self.deconv1 = ConvLayer(128, 64, 3, 2)
        self.norm4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = ConvLayer(64, 32, 3, 2)
        self.norm5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, 3, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        img = self.relu(self.norm1(self.conv1(x)))
        img = self.relu(self.norm2(self.conv2(img)))
        img = self.relu(self.norm3(self.conv3(img)))
        img = self.res1(img)
        img = self.res2(img)
        img = self.res3(img)
        img = self.res4(img)
        img = self.res5(img)
        img = self.relu(self.norm4(self.deconv1(img)))
        img = self.relu(self.norm5(self.deconv2(img)))
        img = self.deconv3(img)
        return img

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.relu(out + residual)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(kernel_size//2),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride) 
        )
    
    def forward(self, x):
        out = self.conv(x)
        return out

class LossNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
   
    def forward(self, x):
        h = self.slice1(x)
        relu1_2 = h
        h = self.slice2(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_3 = h
        h = self.slice4(h)
        relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out
