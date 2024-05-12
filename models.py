import torch
from torch import nn
from torch.nn import ConstantPad2d

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )

    def forward(self, x):
        return self.double_conv(x)

class Conv3BN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3bn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )

    def forward(self, x):
        return self.conv3bn(x)

class AttentionBlock(nn.Module):
    def __init__(self, xin_channels, gin_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.Ax = nn.Sequential(nn.Conv2d(xin_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        self.Ag = nn.Sequential(nn.Conv2d(gin_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        self.psi = nn.Sequential(nn.Conv2d(out_channels, 1, 1),
                nn.BatchNorm2d(1),
                nn.Sigmoid())

    def forward(self, x, g):
        x1 = self.Ax(x)
        g1 = nn.functional.interpolate(self.Ag(g), x1.shape[2:], mode = 'bilinear', align_corners = False)
        a = self.psi(nn.ReLU()(x1 + g1))
        return a * x

class FCN8s(nn.Module):
    def __init__(self, input_channels=1, out_channels=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)

        self.ccov_1 = DoubleConv(input_channels, 64)
        self.ccov_2 = DoubleConv(64, 128)
        self.cov3_3 = Conv3BN(128, 256)
        self.cov3_4 = Conv3BN(256, 512)
        self.cov3_5 = Conv3BN(512, 512)
        self.cov6 = nn.Conv2d(512, 4096, 3, 1, 1)

        self.up3 = nn.Conv2d(256, out_channels, 1)
        self.up4 = nn.Conv2d(512, out_channels, 1)
        self.up6 = nn.ConvTranspose2d(4096, out_channels, 2, 2)
        self.up7 = nn.ConvTranspose2d(out_channels, out_channels, 2, 2)
        self.up8 = nn.ConvTranspose2d(out_channels, out_channels, 8, 8)

    def forward(self, x):
        x1 = self.pool(self.ccov_1(x))
        x2 = self.pool(self.ccov_2(x1))
        x3 = self.pool(self.cov3_3(x2))
        x4 = self.pool(self.cov3_4(x3))
        x5 = self.pool(self.cov3_5(x4))
        x6 = self.cov6(x5)

        x7 = self.up4(x4) + self.up6(x6)
        x8 = self.up3(x3) + self.up7(x7)
        return self.up8(x8)

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)

        self.center_conv = DoubleConv(512, 1024)

        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(512, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)

        x5 = self.center_conv(x4_down)

        x6_up = self.up_1(x5)
        x6c = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(x6c)

        x7_up = self.up_2(x6)
        x7c = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(x7c)

        x8_up = self.up_3(x7)
        x8c = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(x8c)

        x9_up = self.up_4(x8)
        x9c = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(x9c)

        output = self.output(x9)

        return output


class UNetM12(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)

        self.center_conv = DoubleConv(512, 1024)

        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(512, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(256, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)

        x5 = self.center_conv(x4_down)

        x6_up = self.up_1(x5)
        x6 = self.right_conv_1(x6_up)

        x7_up = self.up_2(x6)
        x7 = self.right_conv_2(x7_up)

        x8_up = self.up_3(x7)
        x8c = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(x8c)

        x9_up = self.up_4(x8)
        x9c = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(x9c)

        output = self.output(x9)

        return output


class UNetM34(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)

        self.center_conv = DoubleConv(512, 1024)

        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(512, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(128, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(64, 64)

        self.output = nn.Conv2d(64, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)

        x5 = self.center_conv(x4_down)

        x6_up = self.up_1(x5)
        x6c = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(x6c)

        x7_up = self.up_2(x6)
        x7c = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(x7c)

        x8_up = self.up_3(x7)
        x8 = self.right_conv_3(x8_up)

        x9_up = self.up_4(x8)
        x9 = self.right_conv_4(x9_up)

        output = self.output(x9)

        return output


class UNetM(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)

        self.center_conv = DoubleConv(512, 1024)

        self.up_1 = nn.ConvTranspose2d(1024, 4, 2, 2)
        self.right_conv_1 = DoubleConv(516, 512)
        self.up_2 = nn.ConvTranspose2d(512, 4, 2, 2)
        self.right_conv_2 = DoubleConv(260, 256)
        self.up_3 = nn.ConvTranspose2d(256, 4, 2, 2)
        self.right_conv_3 = DoubleConv(132, 128)
        self.up_4 = nn.ConvTranspose2d(128, 4, 2, 2)
        self.right_conv_4 = DoubleConv(68, 64)

        self.output = nn.Conv2d(64, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)

        x5 = self.center_conv(x4_down)

        x6_up = self.up_1(x5)
        x6c = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(x6c)

        x7_up = self.up_2(x6)
        x7c = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(x7c)

        x8_up = self.up_3(x7)
        x8c = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(x8c)

        x9_up = self.up_4(x8)
        x9c = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(x9c)

        output = self.output(x9)

        return output


class AUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)

        self.center_conv = DoubleConv(512, 1024)

        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.attn_1 = AttentionBlock(512, 1024, 512)
        self.right_conv_1 = DoubleConv(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.attn_2 = AttentionBlock(256, 512, 256)
        self.right_conv_2 = DoubleConv(512, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.attn_3 = AttentionBlock(128, 256, 128)
        self.right_conv_3 = DoubleConv(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.attn_4 = AttentionBlock(64, 128, 64)
        self.right_conv_4 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)

        x5 = self.center_conv(x4_down)

        x6_up = self.up_1(x5)
        x4a = self.attn_1(x4, x5)
        x6c = torch.cat((x6_up, x4a), dim=1)
        x6 = self.right_conv_1(x6c)

        x7_up = self.up_2(x6)
        x3a = self.attn_2(x3, x6)
        x7c = torch.cat((x7_up, x3a), dim=1)
        x7 = self.right_conv_2(x7c)

        x8_up = self.up_3(x7)
        x2a = self.attn_3(x2, x7)
        x8c = torch.cat((x8_up, x2a), dim=1)
        x8 = self.right_conv_3(x8c)

        x9_up = self.up_4(x8)
        x1a = self.attn_4(x1, x8)
        x9c = torch.cat((x9_up, x1a), dim=1)
        x9 = self.right_conv_4(x9c)

        output = self.output(x9)

        return output

class AUNetM12(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)

        self.center_conv = DoubleConv(512, 1024)

        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.attn_1 = AttentionBlock(512, 1024, 512)
        self.right_conv_1 = DoubleConv(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.attn_2 = AttentionBlock(256, 512, 256)
        self.right_conv_2 = DoubleConv(512, 256)
        
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(128, 128)
        
        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(64, 64)

        self.output = nn.Conv2d(64, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)

        x5 = self.center_conv(x4_down)

        x6_up = self.up_1(x5)
        x4a = self.attn_1(x4, x5)
        x6c = torch.cat((x6_up, x4a), dim=1)
        x6 = self.right_conv_1(x6c)

        x7_up = self.up_2(x6)
        x3a = self.attn_2(x3, x6)
        x7c = torch.cat((x7_up, x3a), dim=1)
        x7 = self.right_conv_2(x7c)

        x8_up = self.up_3(x7)
        x8 = self.right_conv_3(x8_up)

        x9_up = self.up_4(x8)
        x9 = self.right_conv_4(x9_up)

        output = self.output(x9)

        return output

class AUNetM34(nn.Module):
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 64)
        self.left_conv_2 = DoubleConv(64, 128)
        self.left_conv_3 = DoubleConv(128, 256)
        self.left_conv_4 = DoubleConv(256, 512)

        self.center_conv = DoubleConv(512, 1024)

        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(512, 512)
        
        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(256, 256)
        
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.attn_3 = AttentionBlock(128, 256, 128)
        self.right_conv_3 = DoubleConv(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.attn_4 = AttentionBlock(64, 128, 64)
        self.right_conv_4 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)
        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)
        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)
        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)

        x5 = self.center_conv(x4_down)

        x6_up = self.up_1(x5)
        x6 = self.right_conv_1(x6_up)

        x7_up = self.up_2(x6)
        x7 = self.right_conv_2(x7_up)

        x8_up = self.up_3(x7)
        x2a = self.attn_3(x2, x7)
        x8c = torch.cat((x8_up, x2a), dim=1)
        x8 = self.right_conv_3(x8c)

        x9_up = self.up_4(x8)
        x1a = self.attn_4(x1, x8)
        x9c = torch.cat((x9_up, x1a), dim=1)
        x9 = self.right_conv_4(x9c)

        output = self.output(x9)

        return output
