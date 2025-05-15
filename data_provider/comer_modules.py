import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from adapter_modules.attention_layer import AttentionLayer

_logger = logging.getLogger(__name__)


# 输入c1(224,64,64), c2(224,32,64), c3(224,16,64)
# c1=Conv1d(c1.T).T, c2=Conv1d(c2.T).T, c3=Conv1d(c3.T).T, 其中Conv1d的in_channel=out_channel=64
# 输出c1(224,64,64), c2(224,32,64), c3(224,16,64)
class DWConv(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=(3,), stride=(1,), padding=(1,), bias=True, groups=dim)

    def forward(self, x, idxs):
        # print('############# DWConv-1')
        # print(x.shape)      # torch.Size([224, 112, 64])

        x1 = x[:, 0:idxs[0], :].contiguous()
        # print(x1.shape)     # torch.Size([224, 64, 64])
        x1 = x1.transpose(1, 2)
        # print(x1.shape)     # torch.Size([224, 64, 64])
        x1 = self.dwconv(x1)
        # print(x1.shape)     # torch.Size([224, 64, 64])

        x2 = x[:, idxs[0]:idxs[1], :].contiguous()
        # print(x2.shape)     # torch.Size([224, 32, 64])
        x2 = x2.transpose(1, 2)
        # print(x2.shape)     # torch.Size([224, 64, 32])
        x2 = self.dwconv(x2)
        # print(x2.shape)     # torch.Size([224, 64, 32])

        x3 = x[:, idxs[1]:, :].contiguous()
        # print(x3.shape)     # torch.Size([224, 16, 64])
        x3 = x3.transpose(1, 2)
        # print(x3.shape)     # torch.Size([224, 64, 16])
        x3 = self.dwconv(x3)
        # print(x3.shape)     # torch.Size([224, 64, 16])

        x = torch.cat([x1, x2, x3], dim=2)
        # print(x.shape)      # torch.Size([224, 64, 112])
        x = x.transpose(1, 2).contiguous()
        # print(x.shape)      # torch.Size([224, 112, 64])
        # print('############# DWConv-2')
        return x


# 输入c1(224,64,32), c2(224,32,32), c3(224,16,32)
# c1,c2,c3=FC(c1,c2,c3), 其中输入和输出节点数为32和64
# 得到c1(224,64,64), c2(224,32,64), c3(224,16,64)
# c1,c2,c3=DWConv(c1,c2,c3),
# 得到c1(224,64,64), c2(224,32,64), c3(224,16,64)
# c1,c2,c3=Dropout(GELU(c1,c2,c3))
# c1,c2,c3=Dropout(FC(c1,c2,c3)), 其中输入和输出节点数为64和32
# 输出c1(224,64,32), c2(224,32,32), c3(224,16,32)
class ConvFFN(nn.Module):
    def __init__(self, in_features=32, hidden_features=64, drop=0.1):
        super().__init__()

        # Layer1
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Layer2
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()

        # Layer3
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, idxs):
        # print('############# ConvFFN-1')
        # print(x.shape)              # torch.Size([224, 112, 32])
        x = self.fc1(x)
        # print(x.shape)              # torch.Size([224, 112, 64])

        # print(type(self.dwconv))    # <class 'Adapter4TS_1D.adapter_modules.comer_modules.DWConv'>
        # print(x.shape)              # torch.Size([224, 112, 64])
        x = self.dwconv(x, idxs)
        x = self.act(x)
        x = self.drop(x)
        # print(x.shape)              # torch.Size([224, 112, 64])

        x = self.fc2(x)
        x = self.drop(x)
        # print(x.shape)              # torch.Size([224, 112, 32])
        # print('############# ConvFFN-2')
        return x


# 输入c1(224,64,192), c2(224,32,192), c3(224,16,192)
# c1=Conv1d(c1.T).T, c2=Conv1d(c2.T).T, c3=Conv1d(c3.T).T, 其中Conv1d的in_channel=out_channel=192
# 得到c1(224,64,192), c2(224,32,192), c3(224,16,192)
# c1=GELU(LayerNorm(c1)), c2=GELU(LayerNorm(c2)), c3=GELU(LayerNorm(c3)),
# 输出c1(224,64,192), c2(224,32,192), c3(224,16,192)
class MultiDWConv(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.dim_half = dim // 2

        self.dwconv = nn.Conv1d(self.dim, self.dim, 3, 1, 1, bias=True, groups=self.dim)

        self.dwconv1 = nn.Conv1d(self.dim_half, self.dim_half, 3, 1, 1, bias=True, groups=self.dim_half)
        self.dwconv2 = nn.Conv1d(self.dim_half, self.dim_half, 5, 1, 2, bias=True, groups=self.dim_half)

        self.dwconv3 = nn.Conv1d(self.dim_half, self.dim_half, 3, 1, 1, bias=True, groups=self.dim_half)
        self.dwconv4 = nn.Conv1d(self.dim_half, self.dim_half, 5, 1, 2, bias=True, groups=self.dim_half)

        self.dwconv5 = nn.Conv1d(self.dim_half, self.dim_half, 3, 1, 1, bias=True, groups=self.dim_half)
        self.dwconv6 = nn.Conv1d(self.dim_half, self.dim_half, 5, 1, 2, bias=True, groups=self.dim_half)

        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(self.dim)

        self.act2 = nn.GELU()
        self.ln2 = nn.LayerNorm(self.dim)

        self.act3 = nn.GELU()
        self.ln3 = nn.LayerNorm(self.dim)

    def forward(self, x, idxs):
        # print('############# MultiDWConv-1')
        # print(x.shape)              # torch.Size([224, 141, 192])

        x1 = x[:, 0:idxs[0], :].contiguous()
        # print(x1.shape)             # torch.Size([224, 19, 192])
        x1 = x1.transpose(1, 2)
        # print(x1.shape)             # torch.Size([224, 192, 19])
        x1 = self.dwconv(x1)
        # print(x1.shape)             # torch.Size([224, 192, 19])

        x2 = x[:, idxs[0]:idxs[1], :].contiguous()
        # print(x2.shape)             # torch.Size([224, 40, 192])
        x2 = x2.transpose(1, 2)
        # print(x2.shape)             # torch.Size([224, 192, 40])
        x2 = self.dwconv(x2)
        # print(x2.shape)             # torch.Size([224, 192, 40])

        x3 = x[:, idxs[1]:, :].contiguous()
        # print(x3.shape)             # torch.Size([224, 82, 192])
        x3 = x3.transpose(1, 2)
        # print(x3.shape)             # torch.Size([224, 192, 82])
        x3 = self.dwconv(x3)
        # print(x3.shape)             # torch.Size([224, 192, 82])

        x11, x12 = x1[:, :x1.shape[1] // 2, :].contiguous(), x1[:, x1.shape[1] // 2:, :].contiguous()
        # print(x11.shape)            # torch.Size([224, 96, 19])
        # print(x12.shape)            # torch.Size([224, 96, 19])
        x11 = self.dwconv1(x11)
        x12 = self.dwconv2(x12)
        # print(x11.shape)            # torch.Size([224, 96, 19])
        # print(x12.shape)            # torch.Size([224, 96, 19])
        x1 = torch.cat([x11, x12], dim=1)
        # print(x1.shape)             # torch.Size([224, 192, 19])
        x1 = self.act1(self.ln1(x1.transpose(1, 2)))
        # print(x1.shape)             # torch.Size([224, 19, 192])

        x21, x22 = x2[:, :x2.shape[1] // 2, :].contiguous(), x2[:, x2.shape[1] // 2:, :].contiguous()
        # print(x21.shape)            # torch.Size([224, 96, 32])
        # print(x22.shape)            # torch.Size([224, 96, 32])
        x21 = self.dwconv3(x21)
        x22 = self.dwconv4(x22)
        # print(x21.shape)            # torch.Size([224, 96, 32])
        # print(x22.shape)            # torch.Size([224, 96, 32])
        x2 = torch.cat([x21, x22], dim=1)
        # print(x2.shape)             # torch.Size([224, 192, 32])
        x2 = self.act2(self.ln2(x2.transpose(1, 2)))
        # print(x2.shape)             # torch.Size([224, 32, 192])

        x31, x32 = x3[:, :x3.shape[1] // 2, :].contiguous(), x3[:, x3.shape[1] // 2:, :].contiguous()
        # print(x31.shape)            # torch.Size([224, 96, 82])
        # print(x32.shape)            # torch.Size([224, 96, 82])
        x31 = self.dwconv5(x31)
        x32 = self.dwconv6(x32)
        # print(x31.shape)            # torch.Size([224, 96, 82])
        # print(x32.shape)            # torch.Size([224, 96, 82])
        x3 = torch.cat([x31, x32], dim=1)
        # print(x3.shape)             # torch.Size([224, 192, 82])
        x3 = self.act3(self.ln3(x3.transpose(1, 2)))
        # print(x3.shape)             # torch.Size([224, 82, 192])

        x = torch.cat([x1, x2, x3], dim=1)
        # print(x.shape)              # torch.Size([224, 141, 192])
        # print('############# MultiDWConv-2')
        return x


# 输入c1(224,64,32), c2(224,32,32), c3(224,16,32)
# c1,c2,c3=FC(c1,c2,c3), 其中输入和输出节点数为32和192
# 得到c1(224,64,192), c2(224,32,192), c3(224,16,192)
# c1,c2,c3=DWConv(c1,c2,c3),
# 得到c1(224,64,192), c2(224,32,192), c3(224,16,192)
# c1,c2,c3=Dropout(GELU(c1,c2,c3))
# c1,c2,c3=Dropout(FC(c1,c2,c3)), 其中输入和输出节点数为192和32
# 输出c1(224,64,32), c2(224,32,32), c3(224,16,32)
class MRFP(nn.Module):
    def __init__(self, in_features=32, hidden_features=192, drop=0.1):
        super().__init__()

        # Layer1
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Layer2
        self.dwconv = MultiDWConv(hidden_features)
        self.act = nn.GELU()

        # Layer3
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, idxs):
        # print('############# MRFP-1')
        # print(x.shape)              # torch.Size([224, 112, 32])
        x = self.fc1(x)

        # print(type(self.dwconv))    # <class 'Adapter4TS_1D.adapter_modules.comer_modules.MultiDWConv'>
        # print(x.shape)              # torch.Size([224, 112, 192])
        x = self.dwconv(x, idxs)
        # print(x.shape)              # torch.Size([224, 112, 192])
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        # print(x.shape)              # torch.Size([224, 112, 32])
        x = self.drop(x)
        # print('############# MRFP-2')
        return x


# 输入c1(224,64,32), c2(224,32,32), c3(224,16,32), 拼接为c(224,112,32)
# 计算tmp=SelfAttn(LayerNorm_query(c),LayerNorm_key(c),LayerNorm_value(c)), c=c+tmp
# 计算tmp=Dropout(ConvFFN(LayerNorm(c))), c=c+tmp
class MultiscaleExtractor(nn.Module):
    def __init__(self, dim=32, num_heads=4, cffn_ratio=2.0, drop=0.1):
        super().__init__()

        # Layer1
        self.attn = AttentionLayer(d_model=dim, n_heads=num_heads)
        self.query_norm = nn.LayerNorm(dim)
        self.keys_norm = nn.LayerNorm(dim)
        self.values_norm = nn.LayerNorm(dim)

        # Layer2
        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop)

    def forward(self, query, key, value, idxs):
        # print('############# MultiscaleExtractor-1')

        # print(type(self.attn))          # <class 'Adapter4TS_1D.adapter_modules.attention_layer.AttentionLayer'>
        # print(type(self.query_norm))    # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(type(self.keys_norm))     # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(type(self.values_norm))   # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(query.shape)              # torch.Size([224, 112, 32])
        # print(key.shape)                # torch.Size([224, 112, 32])
        # print(value.shape)              # torch.Size([224, 112, 32])
        out, _ = self.attn(
            queries=self.query_norm(query),
            keys=self.keys_norm(key),
            values=self.values_norm(value)
        )
        query = query + out
        # print(out.shape)                # torch.Size([224, 112, 32])

        # print(type(self.ffn_norm))      # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(type(self.ffn))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.ConvFFN'>
        # print(type(self.drop_path))     # <class 'torch.nn.modules.dropout.Dropout'>
        # print(query.shape)              # torch.Size([224, 112, 32])
        tmp = self.drop_path(self.ffn(self.ffn_norm(query), idxs))
        # print(tmp.shape)                # torch.Size([224, 112, 32])
        query = query + tmp
        # print('############# MultiscaleExtractor-2')
        return query


# Multi-periodic series to Original series
# 输入x(224,32,32),c(224,112,32)
# 将x拼接到c中, c={c1,c2+x,c3}
# 将x内的全部token之间计算依赖, c=MultiscaleExtractor(c)
class CTI_toC(nn.Module):
    def __init__(self, dim=32, num_heads=4, cffn_ratio=2.0, drop=0.1):
        super().__init__()
        self.cfinter = MultiscaleExtractor(dim=dim, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
    
    def forward(self, x, c, idxs):
        # print('############# CTI_toC-1')
        # print(c.shape)                      # torch.Size([224, 112, 32])
        c1 = c[:, 0:idxs[0], :].contiguous()
        c2 = c[:, idxs[0]:idxs[1], :].contiguous()
        c3 = c[:, idxs[1]:, :].contiguous()
        # print(c1.shape)                     # torch.Size([224, 64, 32])
        # print(c2.shape)                     # torch.Size([224, 32, 32])
        # print(c3.shape)                     # torch.Size([224, 16, 32])
        # print(x.shape)                      # torch.Size([224, 32, 32])
        c2 = c2 + x

        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)                      # torch.Size([224, 112, 32])

        # print(type(self.cfinter))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.MultiscaleExtractor'>
        # print(c.shape)                      # torch.Size([224, 112, 32])
        c = self.cfinter(query=c, key=c, value=c, idxs=idxs)
        # print(c.shape)                      # torch.Size([224, 112, 32])

        # print('############# CTI_toC-2')
        return c


# Multi-periodic series to original series
# 输入x(224,32,32),c(224,112,32)
# 将x拼接到c中, c={c1,c2+x,c3}
# 计算tmp=Dropout(ConvFFN(LayerNorm(c))), c=c+tmp
# 计算x中全部token之间计算依赖, c=MultiscaleExtractor(c)
class Extractor_CTI(nn.Module):
    def __init__(self, dim=32, num_heads=4, cffn_ratio=2.0, drop=0.1):
        super().__init__()

        # Layer1
        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop)

        # Layer2
        self.cfinter = MultiscaleExtractor(dim=dim, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
    
    def forward(self, x, c, idxs):
        # print('############# Extractor_CTI-1')
        # print(c.shape)                          # torch.Size([224, 112, 32])
        c1 = c[:, 0:idxs[0], :].contiguous()
        c2 = c[:, idxs[0]:idxs[1], :].contiguous()
        c3 = c[:, idxs[1]:, :].contiguous()
        # print(c1.shape)                         # torch.Size([224, 64, 32])
        # print(c2.shape)                         # torch.Size([224, 32, 32])
        # print(c3.shape)                         # torch.Size([224, 16, 32])
        # print(x.shape)                          # torch.Size([224, 32, 32])
        c2 = c2 + x

        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)                          # torch.Size([224, 112, 32])

        # print(type(self.ffn_norm))              # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(type(self.ffn))                   # <class 'Adapter4TS_1D.adapter_modules.comer_modules.ConvFFN'>
        # print(type(self.drop_path))             # <class 'torch.nn.modules.dropout.Dropout'>
        # print(c.shape)                          # torch.Size([224, 112, 32])
        tmp = self.drop_path(self.ffn(self.ffn_norm(c), idxs=idxs))
        # print(tmp.shape)                        # torch.Size([224, 112, 32])
        c = c + tmp

        # print(type(self.cfinter))               # <class 'mmseg_custom.models.backbones.comer_modules.MultiscaleExtractor'>
        # print(c.shape)                          # torch.Size([224, 112, 32])
        c = self.cfinter(query=c, key=c, value=c, idxs=idxs)
        # print(c.shape)                          # torch.Size([224, 112, 32])

        # print('############# Extractor_CTI-2')
        return c


# Original series to Multi-periodic series
# 输入x(224,32,32),c(224,112,32)
# 计算tmp=SelfAttn(LayerNorm_query(c),LayerNorm_key(c),LayerNorm_value(c)), c=c+tmp
# 计算tmp=Dropout(ConvFFN(LayerNorm(c))), c=c+tmp
# 将c拼接到x中, x=x+gamma*(c1+c2+c3)
class CTI_toV(nn.Module):
    def __init__(self, dim=32, num_heads=4, cffn_ratio=2.0, drop=0.1, init_values=0.):
        super().__init__()

        # Layer1
        self.attn = AttentionLayer(d_model=dim, n_heads=num_heads)
        self.query_norm = nn.LayerNorm(dim)
        self.keys_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)

        # Layer2
        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop)

        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x, c, idxs):
        # print('############# CTI_toV-1')

        # print(type(self.attn))                  # <class 'Adapter4TS_1D.adapter_modules.attention_layer.AttentionLayer'>
        # print(type(self.query_norm))            # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(type(self.keys_norm))             # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(type(self.value_norm))            # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(c.shape)                          # torch.Size([224, 141, 32])
        c, _ = self.attn(
            queries=self.query_norm(c),
            keys=self.keys_norm(c),
            values=self.value_norm(c)
        )
        # print(c.shape)                          # torch.Size([224, 141, 32])

        # print(type(self.ffn_norm))              # <class 'torch.nn.modules.normalization.LayerNorm'>
        # print(type(self.ffn))                   # <class 'Adapter4TS_1D.adapter_modules.comer_modules.ConvFFN'>
        # print(type(self.drop_path))             # <class 'torch.nn.modules.dropout.Dropout'>
        tmp = self.drop_path(self.ffn(self.ffn_norm(c), idxs=idxs))
        # print(tmp.shape)                        # torch.Size([224, 141, 32])
        c = c + tmp

        # print(c.shape)                          # torch.Size([224, 141, 32])      torch.Size([96, 758, 64])
        c1 = c[:, 0:idxs[0], :].contiguous()
        c2 = c[:, idxs[0]:idxs[1], :].contiguous()
        c3 = c[:, idxs[1]:, :].contiguous()
        # print(c1.shape)                         # torch.Size([224, 19, 32])       torch.Size([96, 107, 64])
        # print(c2.shape)                         # torch.Size([224, 40, 32])       torch.Size([96, 216, 64])
        # print(c3.shape)                         # torch.Size([224, 82, 32])       torch.Size([96, 435, 64])
        c1 = F.interpolate(c1.transpose(1, 2), scale_factor=c2.shape[1]/c1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
        c3 = F.interpolate(c3.transpose(1, 2), scale_factor=c2.shape[1]/c3.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
        # print(c1.shape)                         # torch.Size([224, 40, 32])       torch.Size([96, 215, 64])
        # print(c3.shape)                         # torch.Size([224, 40, 32])       torch.Size([96, 216, 64])
        if c1.shape[1] < c2.shape[1]:
            concat_len = c2.shape[1] - c1.shape[1]
            tmp = c1[:, -1:, :].repeat(1, concat_len, 1)
            c1 = torch.concat([c1, tmp], dim=1)
        if c3.shape[1] < c2.shape[1]:
            concat_len = c2.shape[1] - c3.shape[1]
            tmp = c3[:, -1:, :].repeat(1, concat_len, 1)
            c3 = torch.concat([c3, tmp], dim=1)
        if c1.shape[1] > c2.shape[1]:
            c1 = c1[:, :c2.shape[1], :]
        if c3.shape[1] > c2.shape[1]:
            c3 = c3[:, :c2.shape[1], :]
        # print(c1.shape)                         #                                 torch.Size([96, 216, 64])
        # print(c3.shape)                         #                                 torch.Size([96, 216, 64])

        # print(x.shape)                          # torch.Size([224, 40, 32])       torch.Size([96, 216, 64])
        x = x + self.gamma * (c1 + c2 + c3)
        # print(x.shape)                          # torch.Size([224, 40, 32])       torch.Size([96, 216, 64])
        # print('############# CTI_toV-2')
        return x


class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channel, out_channel):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channel = in_channel
        self.out_channel = out_channel

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(pool_scale),
                    nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=(1,))
                )
            )

    def forward(self, x):
        # print('############# ppm-1')

        # print(x.shape)                      # torch.Size([224, 32, 16])
        # print(self.pool_scales)             # (1, 2, 3, 6)

        ppm_outs = []
        for ppm in self:

            ppm_out = ppm(x)
            # print(type(ppm))                # <class 'torch.nn.modules.container.Sequential'>
            # print(x.shape)                  # torch.Size([224, 32, 16])    torch.Size([224, 32, 16])  torch.Size([224, 32, 16])  torch.Size([224, 32, 16])
            # print(ppm_out.shape)            # torch.Size([224, 32, 1])     torch.Size([224, 32, 2])   torch.Size([224, 32, 3])   torch.Size([224, 32, 6])

            scale_factor = x.shape[-1] / ppm_out.shape[-1]
            upsampled_ppm_out = F.interpolate(ppm_out, scale_factor=scale_factor, mode='linear', align_corners=False, recompute_scale_factor=True)
            # print(upsampled_ppm_out.shape)  # torch.Size([224, 32, 16])    torch.Size([224, 32, 16])  torch.Size([224, 32, 16])  torch.Size([224, 32, 16])
            ppm_outs.append(upsampled_ppm_out)

        # print('############# ppm-2')
        return ppm_outs


# RevIN
# Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift
#   self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)
#   print(x.shape)          # torch.Size([128, 336, 7])
#   x = self.normalize_layer(x, 'norm')
#   print(x.shape)          # torch.Size([128, 336, 7])
#   print(x.shape)          # torch.Size([128, 96, 7])
#   x = self.normalize_layer(x, 'denorm')
#   print(x.shape)          # torch.Size([128, 96, 7])
class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
