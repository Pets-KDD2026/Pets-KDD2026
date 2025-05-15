import math
import torch
from torch import nn
from torch.nn.init import normal_
import torch.nn.functional as F

from models.PatchTST import Model
from adapter_modules.comer_modules import Normalize
from adapter_modules._for_PatchTST import TMPPatchEmbedding, AdapterPatchBlock, DecodeHeadPatch


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims).contiguous()


class FlattenHead(nn.Module):
    def __init__(self, in_feat, out_feat, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(in_feat, out_feat)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# 将整数m划分为n个整数，使得到的整数之间的差值不超过1，结果按照升序排列。
def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n


class ModelAdapter(Model):
    def __init__(
            self,
            configs,                                                        # Backbone独有
            cffn_ratio=2.0, init_values=0., dim_ratio=6.0, period_num=3,    # Adapter独有
    ):
        super().__init__(configs=configs)
        # print('############# PatchTST_Adapter-1')
        # ######################## Original Module

        # print(self.task_name)                       # long_term_forecast
        # print(self.seq_len)                         # 336
        # print(self.pred_len)                        # 336

        # 0. Patch Embedding (Including Position Embedding)
        # print(type(self.patch_embedding))           # <class 'Adapter4TS_1D.layers.Embed.PatchEmbedding'>

        # 1. Encoder Block
        # print(type(self.encoder))                   # <class 'Adapter4TS_1D.layers.Transformer_EncDec.Encoder'>
        # print(len(self.encoder.attn_layers))        # 12
        # print(type(self.encoder.attn_layers[0]))    # <class 'Adapter4TS_1D.layers.Transformer_EncDec.EncoderLayer'>
        # print(self.encoder.conv_layers)             # None
        # print(self.encoder.norm)                    # LayerNorm((32,), eps=1e-05, elementwise_affine=True)

        # 2. Prediction Head
        # d_model*((seq_len-patch_len)//stride+2)
        # print(type(self.head))                      # <class 'Adapter4TS_1D.models.PatchTST.FlattenHead'>
        # print(self.head_nf)                         # 1024
        # print(self.pred_len)                        # 336

        # ######################## Plugin Module

        # 0. 预训练Backbone的模型参数
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.n_heads = configs.n_heads
        self.drop = configs.dropout
        # print(self.d_model)                 # 32
        # print(self.enc_in)                  # 7
        # print(self.n_heads)                 # 4
        # print(self.drop)                    # 0.1

        self.adapter_layer_num = period_num+1
        self.backbone_layer_num = len(self.encoder.attn_layers)
        assert self.backbone_layer_num >= self.adapter_layer_num, f"The Layer Num of Backbone ({self.backbone_layer_num}) is less than Adapter ({self.adapter_layer_num})"
        split_index_list = split_integer(self.backbone_layer_num, self.adapter_layer_num)
        self.interaction_indexes = [[sum(split_index_list[:i]), sum(split_index_list[:i+1])-1] for i in range(self.adapter_layer_num)]
        # print(self.backbone_layer_num)      # 12
        # print(self.adapter_layer_num)       # 4
        # print(split_index_list)             # [3, 3, 3, 3]
        # print(self.interaction_indexes)     # [[0, 2], [3, 5], [6, 8], [9, 11]]

        self.patch_len_1 = configs.patch_len_1
        self.patch_len_2 = configs.patch_len_2
        self.patch_len_3 = configs.patch_len_3
        self.stride_1 = int(self.patch_len_1 / 4)
        self.stride_2 = int(self.patch_len_2 / 4)
        self.stride_3 = int(self.patch_len_3 / 4)
        self.token_num_1 = int((self.seq_len-self.patch_len_1)/self.stride_1 + 2)
        self.token_num_2 = int((self.seq_len-self.patch_len_2)/self.stride_2 + 2)
        self.token_num_3 = int((self.seq_len-self.patch_len_3)/self.stride_3 + 2)
        # print(self.patch_len_1)             # 64
        # print(self.patch_len_2)             # 32
        # print(self.patch_len_3)             # 16
        # print(self.token_num_1)             # 19
        # print(self.token_num_2)             # 40
        # print(self.token_num_3)             # 82

        self.normalize_layer = Normalize(self.enc_in, affine=True, non_norm=False)

        # 1. SPM, 用于生成multi-scale的c
        self.spm = TMPPatchEmbedding(
            d_model=self.d_model,
            patch_len_1=self.patch_len_1, patch_len_2=self.patch_len_2, patch_len_3=self.patch_len_3,
            stride_1=self.stride_1, stride_2=self.stride_2, stride_3=self.stride_3,
            patch_num_1=self.token_num_1, patch_num_2=self.token_num_2, patch_num_3=self.token_num_3,
        )
        self.spm.apply(self._init_weights)
        # print(type(self.spm))               # <class 'Adapter4TS_1D.adapter_modules.comer_modules.CNN'>
        # print(self.d_model)                 # 32

        # 2. Multi-Level Embedding, 用于为multi-scale的c添加层级嵌入信息
        self.level_embed = nn.Parameter(torch.zeros(period_num, self.d_model))
        normal_(self.level_embed)
        # print(self.level_embed.shape)       # torch.Size([3, 32])
        # print(self.d_model)                 # 32

        # 3. 基于BackboneBlock封装得到的AdapterBlock, 其中负责将c和x融合，并
        self.interactions = nn.Sequential(*[
            AdapterPatchBlock(
                dim=self.d_model, num_heads=self.n_heads, cffn_ratio=cffn_ratio, drop=self.drop,
                init_values=init_values, dim_ratio=dim_ratio, extra_CTI=(True if i == len(self.interaction_indexes) - 1 else False)
            )
            for i in range(len(self.interaction_indexes))
        ])
        self.interactions.apply(self._init_weights)
        # print(type(AdapterBlock))           # <class 'type'>
        # print(self.d_model)                 # 32
        # print(self.n_heads)                 # 4
        # print(cffn_ratio)                   # 0.25
        # print(self.drop)                    # 0.1
        # print(init_values)                  # 0.0
        # print(dim_ratio)                    # 6.0
        # print(self.interaction_indexes)     # [[0, 2], [3, 5], [6, 8], [9, 11]]

        # 4. Decode Head
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.flatten = nn.Flatten(start_dim=-2)
            self.up_xt = nn.Linear(in_features=int(self.token_num_2 * self.d_model), out_features=self.pred_len, bias=True)
            self.up_xt.apply(self._init_weights)
            self.up_ct = nn.Linear(in_features=self.seq_len, out_features=self.pred_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.head_ppm = DecodeHeadPatch(token_num_max=self.token_num_3, d_model=self.d_model, pred_len=self.pred_len)

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.flatten = nn.Flatten(start_dim=-2)
            self.up_xt = nn.Linear(in_features=int(self.token_num_2 * self.d_model), out_features=self.seq_len, bias=True)
            self.up_xt.apply(self._init_weights)
            self.up_ct = nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.head_ppm = DecodeHeadPatch(token_num_max=self.token_num_3, d_model=self.d_model, pred_len=self.seq_len)

        elif self.task_name == 'classification':
            self.normt = nn.LayerNorm(self.d_model)
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                in_features=(self.token_num_1 + self.token_num_2*2 + self.token_num_3)*self.d_model*self.enc_in,
                out_features=configs.num_class
            )

        # print('############# PatchTST_Adapter-2')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print('############# PatchTST_Adapter.forecast-1')

        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x_enc = self.normalize_layer(x_enc, 'norm')
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 7, 336])

        # 1. SPM forward
        # SPM即Spatial Pyramid Matching, 是一种利用空间金字塔进行图像匹配、识别、分类的算法, SPM在不同分辨率上统计图像特征点分布，从而获取图像的局部信息。
        # print(type(self.spm))               # <class 'Adapter4TS_1D.adapter_modules.comer_modules.TMPEmbedding'>
        ct, c1, c2, c3 = self.spm(x_enc)
        # print(ct.shape)                     # torch.Size([224, 336])
        # print(c1.shape)                     # torch.Size([224, 19, 32])
        # print(c2.shape)                     # torch.Size([224, 40, 32])
        # print(c3.shape)                     # torch.Size([224, 82, 32])

        # 2. Multi-Level Embedding
        # print(self.level_embed[0].shape)    # torch.Size([32])
        # print(self.level_embed[1].shape)    # torch.Size([32])
        # print(self.level_embed[2].shape)    # torch.Size([32])
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        # print(c1.shape)                     # torch.Size([224, 19, 32])
        # print(c2.shape)                     # torch.Size([224, 40, 32])
        # print(c3.shape)                     # torch.Size([224, 82, 32])
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)                      # torch.Size([224, 141, 32])

        # 3. Patch & Position Embedding
        # print(type(self.patch_embedding))   # <class 'Adapter4TS_1D.layers.Embed.PatchEmbedding'>
        enc_out, n_vars = self.patch_embedding(x_enc)
        # print(enc_out.shape)                # torch.Size([224, 40, 32])
        # print(n_vars)                       # 7

        # 4. Backbone & Adapter
        outs = []
        # print(len(self.interactions))       # 4
        # print(len(self.encoder.attn_layers))# 12
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            # print(indexes)                  # [0, 2]            [3, 5]              [6, 8]                  [9, 11]
            adapter_block = self.interactions[i]
            backbone_blocks = self.encoder.attn_layers[indexes[0]:indexes[-1] + 1]
            # print(type(adapter_block))      # <class 'Adapter4TS_1D.adapter_modules.comer_modules.AdapterBlock'>
            # print(len(backbone_blocks))     # 3
            # print(type(backbone_blocks[0])) # <class 'Adapter4TS_1D.layers.Transformer_EncDec.EncoderLayer'>
            # print(enc_out.shape)            # torch.Size([224, 40, 32])
            # print(c.shape)                  # torch.Size([224, 141, 32])
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, idxs=[c1.shape[1], c1.shape[1]+c2.shape[1]])
            # print(enc_out.shape)            # torch.Size([224, 40, 32])
            # print(c.shape)                  # torch.Size([224, 141, 32])
            outs.append(enc_out)

        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([224, 40, 32])
        # print(x1.shape)                     # torch.Size([224, 40, 32])
        # print(x2.shape)                     # torch.Size([224, 40, 32])
        # print(x3.shape)                     # torch.Size([224, 40, 32])

        # 5.1 Split Multi-period condition (MP-cond)
        # print(c.shape)                      # torch.Size([224, 141, 32])
        c1 = c[:, 0:c1.shape[1], :]
        c2 = c[:, c1.shape[1]:c1.shape[1]+c2.shape[1], :]
        c3 = c[:, c1.shape[1]+c2.shape[1]:, :]
        # print(c1.shape)                     # torch.Size([224, 19, 32])
        # print(c2.shape)                     # torch.Size([224, 40, 32])
        # print(c3.shape)                     # torch.Size([224, 82, 32])

        # 5.2 Fusion Multi-scale hidden feature from Adapter to Trend and Cond1,2,3
        x1 = F.interpolate(x1.transpose(1, 2).contiguous(), scale_factor=c1.shape[1]/x1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        x3 = F.interpolate(x3.transpose(1, 2).contiguous(), scale_factor=c3.shape[1]/x3.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        f1 = self.norm1(c1+x1)
        f2 = self.norm2(c2+x2)
        f3 = self.norm3(c3+x3)
        # print(f1.shape)                     # torch.Size([224, 19, 32])
        # print(f2.shape)                     # torch.Size([224, 40, 32])
        # print(f3.shape)                     # torch.Size([224, 82, 32])

        # 6.1 Down-to-Up Path Decoder
        dec_out = self.head_ppm([f1, f2, f3])
        # print(type(self.head_ppm))          # <class 'Adapter4TS_1D.adapter_modules.comer_modules.DecodeHead'>
        # print(dec_out.shape)                # torch.Size([224, 96])

        # 6.2 TrendResid AutoRegression
        # print(ct.shape)                     # torch.Size([224, 336])
        ct = self.up_ct(ct)
        # print(ct.shape)                     # torch.Size([224, 96])

        # 6.3 Fusion
        # print(xt.shape)                     # torch.Size([224, 40, 32])
        xt = self.up_xt(self.flatten(xt))
        # print(xt.shape)                     # torch.Size([224, 96])
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([224, 96])

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-1])).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 7, 96])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 96, 7])
        dec_out = self.normalize_layer(dec_out, 'denorm')
        # print(dec_out.shape)                # torch.Size([32, 96, 7])

        # print('############# PatchTST_Adapter.forecast-2')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        def avgImputation(x, mask):
            x_imp = x

            for i in range(x.shape[0]):
                for j in range(x.shape[2]):
                    seq = x[i, :, j]
                    seq_mask = mask[i, :, j]

                    # 首先需要确保seq[-1]非零
                    if seq_mask[-1] == 0:
                        idx = seq_mask.shape[0] - 1
                        while seq_mask[idx] == 0:
                            idx -= 1
                        seq[-1] = seq[idx]

                    # seq_mask[start,end-1]是连续为0的一段序列
                    start = 0
                    while start < seq_mask.shape[0]:
                        if seq_mask[start] == 1:
                            start += 1
                        else:
                            for end in range(start, seq_mask.shape[0]):
                                if seq_mask[end] == 1:
                                    break
                            step = (seq[end] - seq[start - 1]) / (end - start + 1)
                            for shift in range(end - start):
                                seq_imp = seq[start - 1] + (shift + 1) * step
                                x_imp[i, start + shift, j] = seq_imp
                            start = end + 1

            return x_imp

        # print('############# PatchTST_Adapter.imputation-1')
        # Normalization from Non-stationary Transformer
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        # print(mask.shape)                   # torch.Size([32, 336, 7])
        # 使用avgImputation可以普遍改善Timesformer在所有数据集上的性能
        x_enc = avgImputation(x_enc, mask)
        x_enc = self.normalize_layer(x_enc, 'norm')

        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 7, 336])
        ct, c1, c2, c3 = self.spm(x_enc)
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        c = torch.cat([c1, c2, c3], dim=1)
        # print(ct.shape)                     # torch.Size([224, 336])
        # print(c1.shape)                     # torch.Size([224, 19, 32])
        # print(c2.shape)                     # torch.Size([224, 40, 32])
        # print(c3.shape)                     # torch.Size([224, 82, 32])
        # print(c.shape)                      # torch.Size([224, 141, 32])

        # print(x_enc.shape)                  # torch.Size([32, 7, 336])
        enc_out, n_vars = self.patch_embedding(x_enc)
        # print(enc_out.shape)                # torch.Size([224, 40, 32])
        # print(n_vars)                       # 7

        outs = []
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.encoder.attn_layers[indexes[0]:indexes[-1] + 1]
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, idxs=[c1.shape[1], c1.shape[1] + c2.shape[1]])
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([224, 40, 32])
        # print(x1.shape)                     # torch.Size([224, 40, 32])
        # print(x2.shape)                     # torch.Size([224, 40, 32])
        # print(x3.shape)                     # torch.Size([224, 40, 32])

        c1 = c[:, 0:c1.shape[1], :]
        c2 = c[:, c1.shape[1]:c1.shape[1] + c2.shape[1], :]
        c3 = c[:, c1.shape[1] + c2.shape[1]:, :]
        # print(c1.shape)                     # torch.Size([224, 19, 32])
        # print(c2.shape)                     # torch.Size([224, 40, 32])
        # print(c3.shape)                     # torch.Size([224, 82, 32])

        x1 = F.interpolate(x1.transpose(1, 2).contiguous(), scale_factor=c1.shape[1] / x1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        x3 = F.interpolate(x3.transpose(1, 2).contiguous(), scale_factor=c3.shape[1] / x3.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([224, 19, 32])
        # print(f2.shape)                     # torch.Size([224, 40, 32])
        # print(f3.shape)                     # torch.Size([224, 82, 32])

        dec_out = self.head_ppm([f1, f2, f3])
        ct = self.up_ct(ct)
        xt = self.up_xt(self.flatten(xt))
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([224, 336])

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-1])).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 7, 336])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layer(dec_out, 'denorm')
        # print(dec_out.shape)                # torch.Size([32, 336, 7])
        # print('############# PatchTST_Adapter.imputation-2')
        return dec_out

    def anomaly_detection(self, x_enc):
        # print('############# PatchTST_Adapter.anomaly_detection-1')
        # print(x_enc.shape)                  # torch.Size([32, 100, 55])
        x_enc = self.normalize_layer(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 55, 100])
        ct, c1, c2, c3 = self.spm(x_enc)
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        c = torch.cat([c1, c2, c3], dim=1)
        # print(ct.shape)                     # torch.Size([1760, 100])
        # print(c1.shape)                     # torch.Size([1760, 4, 32])
        # print(c2.shape)                     # torch.Size([1760, 10, 32])
        # print(c3.shape)                     # torch.Size([1760, 23, 32])
        # print(c.shape)                      # torch.Size([1760, 37, 32])

        # print(x_enc.shape)                  # torch.Size([32, 55, 100])
        enc_out, n_vars = self.patch_embedding(x_enc)
        # print(enc_out.shape)                # torch.Size([1760, 10, 32])
        # print(n_vars)                       # 55

        outs = []
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.encoder.attn_layers[indexes[0]:indexes[-1] + 1]
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, idxs=[c1.shape[1], c1.shape[1] + c2.shape[1]])
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([1760, 10, 32])
        # print(x1.shape)                     # torch.Size([1760, 10, 32])
        # print(x2.shape)                     # torch.Size([1760, 10, 32])
        # print(x3.shape)                     # torch.Size([1760, 10, 32])

        c1 = c[:, 0:c1.shape[1], :]
        c2 = c[:, c1.shape[1]:c1.shape[1] + c2.shape[1], :]
        c3 = c[:, c1.shape[1] + c2.shape[1]:, :]
        # print(c1.shape)                     # torch.Size([1760, 4, 32])
        # print(c2.shape)                     # torch.Size([1760, 10, 32])
        # print(c3.shape)                     # torch.Size([1760, 23, 32])

        x1 = F.interpolate(x1.transpose(1, 2).contiguous(), scale_factor=c1.shape[1] / x1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        x3 = F.interpolate(x3.transpose(1, 2).contiguous(), scale_factor=c3.shape[1] / x3.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([1760, 4, 32])
        # print(f2.shape)                     # torch.Size([1760, 10, 32])
        # print(f3.shape)                     # torch.Size([1760, 23, 32])

        dec_out = self.head_ppm([f1, f2, f3])
        ct = self.up_ct(ct)
        xt = self.up_xt(self.flatten(xt))
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([1760, 100])

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-1])).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 55, 100])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layer(dec_out, 'denorm')
        # print(dec_out.shape)                # torch.Size([32, 100, 55])
        # print('############# PatchTST_Adapter.anomaly_detection-2')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # print('############# PatchTST_Adapter.classification-1')
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])          torch.Size([32, 1751, 3])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 7, 336])          torch.Size([32, 3, 1751])
        ct, c1, c2, c3 = self.spm(x_enc)
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        c = torch.cat([c1, c2, c3], dim=1)
        # print(ct.shape)                     # torch.Size([224, 336])            torch.Size([96, 1751])
        # print(c1.shape)                     # torch.Size([224, 19, 32])         torch.Size([96, 107, 32])
        # print(c2.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(c3.shape)                     # torch.Size([224, 82, 32])         torch.Size([96, 435, 32])
        # print(c.shape)                      # torch.Size([224, 141, 32])        torch.Size([96, 758, 32])

        # print(x_enc.shape)                  # torch.Size([32, 7, 336])          torch.Size([32, 3, 1751])
        enc_out, n_vars = self.patch_embedding(x_enc)
        # print(enc_out.shape)                # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(n_vars)                       # 7                                 3

        outs = []
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.encoder.attn_layers[indexes[0]:indexes[-1] + 1]
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, idxs=[c1.shape[1], c1.shape[1] + c2.shape[1]])
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(x1.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(x2.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(x3.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])

        c1 = c[:, 0:c1.shape[1], :]
        c2 = c[:, c1.shape[1]:c1.shape[1] + c2.shape[1], :]
        c3 = c[:, c1.shape[1] + c2.shape[1]:, :]
        # print(c1.shape)                     # torch.Size([224, 19, 32])         torch.Size([96, 107, 32])
        # print(c2.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(c3.shape)                     # torch.Size([224, 82, 32])         torch.Size([96, 435, 32])

        x1 = F.interpolate(x1.transpose(1, 2).contiguous(), scale_factor=c1.shape[1] / x1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        x3 = F.interpolate(x3.transpose(1, 2).contiguous(), scale_factor=c3.shape[1] / x3.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([224, 19, 32])         torch.Size([96, 107, 32])
        # print(f2.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(f3.shape)                     # torch.Size([224, 82, 32])         torch.Size([96, 435, 32])

        # Concatenate
        # print(ct.shape)                     # torch.Size([224, 336])            torch.Size([96, 1751])
        ct = torch.reshape(ct, (-1, n_vars, ct.shape[-1])).contiguous()
        # print(ct.shape)                     # torch.Size([32, 7, 336])          torch.Size([32, 3, 1751])
        ct, _ = self.patch_embedding(ct)
        # print(ct.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        # print(xt.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        ft = self.normt(ct + xt)
        # print(ft.shape)                     # torch.Size([224, 40, 32])         torch.Size([96, 216, 32])
        output = torch.concat([ft, f1, f2, f3], dim=1)
        # print(output.shape)                 # torch.Size([224, 221, 32])        torch.Size([96, 1190, 32])
        output = torch.reshape(output, (-1, n_vars, output.shape[-2], output.shape[-1])).contiguous()
        # print(output.shape)                 # torch.Size([32, 7, 221, 32])      torch.Size([32, 3, 1190, 32])
        output = output.permute(0, 1, 3, 2).contiguous()
        # print(output.shape)                 # torch.Size([32, 7, 32, 221])      torch.Size([32, 3, 32, 1190])
        output = self.flatten(output)
        # print(output.shape)                 # torch.Size([32, 7, 7072])         torch.Size([32, 3, 76160])

        # Reshape & Classifier
        output = self.act(output)
        output = self.dropout(output)
        # print(output.shape)                 # torch.Size([32, 7, 7072])         torch.Size([32, 3, 76160])
        output = output.reshape(output.shape[0], -1).contiguous()
        # print(output.shape)                 # torch.Size([32, 49504])           torch.Size([32, 228480])
        output = self.projection(output)
        # print(output.shape)                 # torch.Size([32, 4])               torch.Size([32, 4])

        # print('############# PatchTST_Adapter.classification-2')
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

