import torch
import torch.nn as nn
import einops


class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm=nn.InstanceNorm3d):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class HybridConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv2d = DoubleConv2D(in_channels*2, out_channels)
        self.conv3d = DoubleConv3D(in_channels*2, out_channels)

    def forward(self, x2, x3):
        # x2 => 2d, x3 => 3d
        d2 = torch.cat([x2, einops.rearrange(x3, '1 C N H W -> N C H W')], dim=1)
        d3 = torch.cat([x3, einops.rearrange(x2, 'B C H W -> 1 C B H W')], dim=1)

        y2 = self.conv2d(d2)
        y3 = self.conv3d(d3)

        return y2, y3
    

class HybridCat3D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv3d = DoubleConv3D(in_channels*2, out_channels)

    def forward(self, x2, x3):
        # x2 => 2d, x3 => 3d
        d3 = torch.cat([x3, einops.rearrange(x2, 'B C H W -> 1 C B H W')], dim=1)
        y3 = self.conv3d(d3)
        return y3


class SpatialAttention2D(nn.Module):
    def __init__(self, k=(5, 5), p=(2, 2)):
        super(SpatialAttention2D, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y) * x
    

class SpatialAttention3D(nn.Module):
    def __init__(self, k=(3, 5, 5), p=(1, 2, 2)):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=k, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y) * x
    

class HybridDown(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.hybird_net = HybridConv(in_channels, out_channels)
        self.s_atten2d = SpatialAttention2D()
        self.s_atten3d = SpatialAttention3D()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x2, x3):
        y2, y3 = self.hybird_net(x2, x3)
        y2 = self.s_atten2d(y2)
        y3 = self.s_atten3d(y3)
        y2 = self.max_pool2d(y2)
        y3 = self.max_pool3d(y3)
        return y2, y3


class CatUp2d(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(2*in_channels, 2*in_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConv2D(2*in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(self.up(x))
    

class CatUp3d(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(2*in_channels, 2*in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv3D(2*in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(self.up(x))
    

class HybridCatUp(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels*3, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv3D(out_channels, out_channels)
    
    def forward(self, x2, x3, _x3):
        x2 = einops.rearrange(x2, 'B C H W -> 1 C B H W')
        x = torch.concat([x2, x3, _x3], dim=1)
        return self.conv(self.up(x))


class TriHybridUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=16, auxiliary=True) -> None:
        """
        auxiliary: the auxiliary decoder
        """
        super().__init__()

        self.auxiliary = auxiliary

        self.inc2d = DoubleConv2D(in_channels, features)
        self.inc3d = DoubleConv3D(in_channels, features)

        self.d1 = HybridDown(1*features, 2*features)
        self.d2 = HybridDown(2*features, 4*features)
        self.d3 = HybridDown(4*features, 8*features)

        self.hybrid_concat = HybridCat3D(8*features, 8*features)
        self.u_hybrid_1 = HybridCatUp(8*features, 4*features)
        self.u_hybrid_2 = HybridCatUp(4*features, 2*features)
        self.u_hybrid_3 = HybridCatUp(2*features, 1*features)

        self.conv2d = DoubleConv2D(8*features, 8*features)
        self.u_2d_1 = CatUp2d(8 * features, 4 * features)
        self.u_2d_2 = CatUp2d(4 * features, 2 * features)
        self.u_2d_3 = CatUp2d(2 * features, 1 * features)

        self.conv3d = DoubleConv3D(8*features, 8*features)
        self.u_3d_1 = CatUp3d(8 * features, 4 * features)
        self.u_3d_2 = CatUp3d(4 * features, 2 * features)
        self.u_3d_3 = CatUp3d(2 * features, 1 * features)

        self.outc_hybrid = nn.Conv3d(features, out_channels, kernel_size=1)
        self.outc_2d = nn.Conv2d(features, 2, kernel_size=1)
        self.outc_3d = nn.Conv3d(features, 2, kernel_size=1)
    
    def forward(self, x2, x3):
        y2_0 = self.inc2d(x2)
        y3_0 = self.inc3d(x3)

        y2_1, y3_1 = self.d1(y2_0, y3_0)
        y2_2, y3_2 = self.d2(y2_1, y3_1)
        y2_3, y3_3 = self.d3(y2_2, y3_2)

        o_hybrid = self.hybrid_concat(y2_3, y3_3)
        o_hybrid = self.u_hybrid_1(y2_3, y3_3, o_hybrid)
        o_hybrid = self.u_hybrid_2(y2_2, y3_2, o_hybrid)
        o_hybrid = self.u_hybrid_3(y2_1, y3_1, o_hybrid)
        o_hybrid = self.outc_hybrid(o_hybrid)

        if self.auxiliary:
            o_2d = self.conv2d(y2_3)
            o_2d = self.u_2d_1(y2_3, o_2d)
            o_2d = self.u_2d_2(y2_2, o_2d)
            o_2d = self.u_2d_3(y2_1, o_2d)
            o_2d = self.outc_2d(o_2d)

            o_3d = self.conv3d(y3_3)
            o_3d = self.u_3d_1(y3_3, o_3d)
            o_3d = self.u_3d_2(y3_2, o_3d)
            o_3d = self.u_3d_3(y3_1, o_3d)
            o_3d = self.outc_3d(o_3d)
        else:
            o_2d = None
            o_3d = None

        return o_hybrid, o_2d, o_3d
