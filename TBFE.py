import torch
import torch.nn as nn

class TBFE1D(nn.Module):
    """
    把 TBFE 改造成适合 (B,C,1,W) 狭长特征图
    输入: (B,C,1,W)
    输出: (B,2*reduction_N,1,W)
    """
    def __init__(self, in_channels, reduction_N=32):
        super().__init__()
        self.point_wise = nn.Conv2d(in_channels, reduction_N, kernel_size=1, bias=False)
        self.depth_wise = nn.Sequential(
            nn.Conv2d(reduction_N, reduction_N, kernel_size=(1,3), padding=(0,1), groups=reduction_N, bias=False),
            nn.BatchNorm2d(reduction_N),
            nn.ReLU(inplace=True)
        )
        # 3D卷积退化为 (1,1,3) 的3D卷积，但这里用 1×3 代替即可
        self.conv3d_like = nn.Conv2d(reduction_N, reduction_N, kernel_size=(1,3),
                                     padding=(0,1), groups=reduction_N, bias=False)
        self.bn = nn.BatchNorm2d(2 * reduction_N)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B,C,1,W)
        x1 = self.point_wise(x)                # (B,reduction_N,1,W)
        x2 = self.depth_wise(x1)               # (B,reduction_N,1,W)
        x2 = x1 + x2                           # 残差
        x3 = self.conv3d_like(x1)              # (B,reduction_N,1,W)
        out = torch.cat([x2, x3], dim=1)       # (B,2*reduction_N,1,W)
        out = self.relu(self.bn(out))
        return out

class MicroResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c//4, 1), nn.BatchNorm2d(c//4), nn.ReLU(),
            nn.Conv2d(c//4, c//4, (1,3), padding=(0,1), groups=c//4), nn.BatchNorm2d(c//4), nn.ReLU(),
            nn.Conv2d(c//4, c, 1), nn.BatchNorm2d(c)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv(x) + x)

class TBFE1DPlus(nn.Module):
    def __init__(self, in_channels, reduction_N=64):
        super().__init__()
        self.tbfe = TBFE1D(in_channels, reduction_N)
        self.res1 = MicroResBlock(2*reduction_N)
        self.res2 = MicroResBlock(2*reduction_N)

    def forward(self, x):
        x = self.tbfe(x)        # (B,128,1,W)
        x = self.res1(x)
        x = self.res2(x)
        return x