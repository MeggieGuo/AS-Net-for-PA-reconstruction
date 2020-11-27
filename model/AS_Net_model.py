import argparse
import torch
from .AS_Net_parts import *
from .Res_attn import Bottleneck
from .context_block import ContextBlock



class AS_Net(nn.Module):
    def __init__(self, in_channels=20, up_mode='transpose'):
        super(AS_Net, self).__init__()
        self.up_mode = up_mode
        self.in_channels = in_channels

        self.down1 = DownConvNN(in_channels, 32)
        self.down2 = DownConvNN(32, 64)
        self.down3 = DownConvNN(64, 128)
        self.attn3 = Self_Attn(128, 'relu')
        self.down4 = DownConvNN(128, 256)
        self.attn4 = Self_Attn(256, 'relu')
        self.bottom0 = BottomNN(256)

        self.bottom1 = Atrousblock(256)

        self.up1 = UpConv(320, 128)
        self.up2 = UpConv(144, 64)
        self.up3 = UpConv(80, 32)
        self.up4 = UpConv(48, 16)
        self.sidelayer = Sidelayer(16)
        self.sidepool = SidePool()
        self.bfpath1 = ConvBlock_keb(1, 64)

        self.bfpath2 = ConvBlock_keb(64, 128)
        self.bfattn1 = Bottleneck(128, 32, 0.6)

        self.bfpath3 = ConvBlock_keb(128, 256)
        self.bfattn2 = Bottleneck(256, 64, 0.4)

        self.bfpath4 = ConvBlock_keb(256, 512)
        self.bfattn3 = Bottleneck(512, 128, 0.5)

        self.bfpath5 = ConvBlock_keb(512, 256)
        self.bfattn4 = Bottleneck(256, 64, 0.5)

        self.bfpath6 = ConvBlock_keb(256, 16)
        self.bfattn5 = Bottleneck(16, 4, 0.5)

        self.flayer2 = Featurelayer(16, 1)
        self.psp = PSPblock()

        self.fusion_glb = ContextBlock(32, 0.5)

        self.final1 = FinConv(32, 128)
        self.final2 = FinConv(128, 1)

    def forward(self, x, bfimg):
        # SFE
        bfimg = self.bfpath1(bfimg)

        bfimg = self.bfpath2(bfimg)
        bfimg = self.bfattn1(bfimg)

        bfimg = self.bfpath3(bfimg)
        bfimg = self.bfattn2(bfimg)

        bfimg = self.bfpath4(bfimg)
        bfimg = self.bfattn3(bfimg)

        bfimg = self.bfpath5(bfimg)
        bfimg = self.bfattn4(bfimg)

        bfimg = self.bfpath6(bfimg)
        bfimg = self.bfattn5(bfimg)

        bf_feature = self.flayer2(bfimg)
        ff = self.psp(bfimg)

        # Encoder
        out = self.down1(x)
        out = self.down2(out)
        out = self.down3(out)
        out = self.attn3(out)
        out = self.down4(out)
        out = self.attn4(out)
        out = self.bottom0(out)
        out = self.bottom1(out)
        out = torch.cat((ff, out), 1)

        # Decoder
        out = self.up1(out)
        out = self.sidepool(bfimg, out)
        out = self.up2(out)
        out = self.sidepool(bfimg, out)
        out = self.up3(out)
        out = self.sidepool(bfimg, out)
        out = self.up4(out)
        side_out = self.sidelayer(out)

        #FF
        out = torch.cat((bfimg, out), 1)
        out = self.fusion_glb(out)
        out = self.final1(out)
        out = self.final2(out)

        return out, bf_feature, side_out


