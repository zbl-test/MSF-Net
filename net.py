import torch, math
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 1. 基础模块 ----------------
class ByteAug(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training: return x
        mask = torch.rand_like(x, dtype=torch.float) > self.p
        x = x * mask
        noise = torch.randn_like(x) * 3
        return torch.clamp(x + noise, 0, 255).long()

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1), nn.ReLU(),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid())
    def forward(self, x): return x * self.att(x)

class MBConv(nn.Module):
    def __init__(self, in_c, out_c, ks=3, se=True, drop_path=0.):
        super().__init__()
        hidden = 4 * in_c
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, (1,ks), padding=(0,ks//2),
                      groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(),
            SE(hidden) if se else nn.Identity(),
            nn.Conv2d(hidden, out_c, 1, bias=False), nn.BatchNorm2d(out_c)
        )
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.skip = nn.Identity() if in_c==out_c else nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        return self.skip(x) + self.drop_path(self.conv(x))

class DropPath(nn.Module):
    def __init__(self, drop_prob): super().__init__(); self.drop_prob=drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob==0: return x
        keep = 1-self.drop_prob
        shape = (x.size(0),) + (1,)*(x.ndim-1)
        rand = keep + torch.rand(shape, device=x.device)
        return x.div(keep) * rand.floor()

# ---------------- 2. Patch Attention ----------------
class PatchAttn(nn.Module):
    def __init__(self, d_model=192, nhead=8, layers=2):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1)
        self.tf = nn.TransformerEncoder(encoder, layers)
    def forward(self, x):          # (B,C,H,W)  例 (B,192,1,64)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)   # (B, HW, C)
        x = x + self.tf(x)
        return x.mean(1)                         # 全局池化

# ---------------- 3. DCT 频率分支 ----------------
class DCTBranch(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        self.dct = nn.Conv1d(1, 64, 8, stride=8, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(64 * 64, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, out))
    def forward(self, x):          # (B, 512)
        x = x.float()
        x = x.unsqueeze(1)         # (B,1,512)
        x = self.dct(x)            # (B,64,64)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

# ------------- 3. 熵分支 -------------
class EntBranch(nn.Module):
    def __init__(self, out=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(257, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, out))
    def forward(self, x):
        B=x.size(0)
        x=x.float()/255
        h=torch.stack([torch.histc(x[i],bins=256,min=0,max=1)+1e-6 for i in range(B)])
        h=h/h.sum(1,keepdim=True)
        ent=-(h*h.log()).sum(1,keepdim=True)
        return self.mlp(torch.cat([h,ent],1))

# ---------------- 4. 分支结构 ----------------
class Branch(nn.Module):
    def __init__(self, d, ch):
        super().__init__()
        H = d * 8
        self.stem = nn.Conv2d(1, ch, (H, 1))
        self.stage = nn.Sequential(
            MBConv(ch, ch * 2, 5, se=True, drop_path=0.1),
            MBConv(ch * 2, ch * 2, 5, se=True, drop_path=0.15),
            MBConv(ch * 2, ch * 4, 5, se=True, drop_path=0.2),
            PatchAttn(ch * 4, nhead=8, layers=2)   # 替换全局池化
        )
    def forward(self, x):
        return self.stage(self.stem(x))

# ---------------- 5. 最终模型（无拒识） ----------------
class FileFormatAnalyzer(nn.Module):
    def __init__(self, num_classes=75, dim=[4, 8, 16, 32], ch=192):
        super().__init__()
        self.aug = ByteAug(0.2)
        self.ngram = nn.ModuleDict()
        for d in dim:
            self.ngram[str(d)] = Branch(d, ch)

        self.byte = nn.Sequential(
            nn.Linear(512, ch * 2), nn.ReLU(), nn.Dropout(0.3))
        self.ent = EntBranch(out=ch * 2)
        self.dct = DCTBranch(out=ch)      # 频率分支

        total_feat = len(dim) * ch * 4 + ch * 2 + ch * 2 + ch
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_feat, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, num_classes))  # 普通线性头

    def forward(self, ngram, raw):
        raw = self.aug(raw)
        n_out = torch.cat([self.ngram[str(d)](ngram[str(d)]) for d in self.ngram], 1)
        b_out = self.byte(raw.float() / 255)
        e_out = self.ent(raw)
        d_out = self.dct(raw)
        feat = torch.cat([n_out, b_out, e_out, d_out], 1)
        return self.head(feat)