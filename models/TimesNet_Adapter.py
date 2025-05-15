import math
import torch
from torch import nn
from torch.nn.init import normal_
import torch.nn.functional as F

from models.PatchTST_Adapter import split_integer
from models.TimesNet import Model
from adapter_modules.comer_modules import Normalize
from adapter_modules._for_TimesNet import TMPTemporalEmbedding, AdapterTemporalBlock, DecodeHeadTemporal


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('############# Inception_Block_V1-1')

        # print(self.num_kernels)             # 6                                   6
        res_list = []
        for i in range(self.num_kernels):
            # print(i)                        # 0,1,2,3,4,5                         0,1,2,3,4,5
            tmp = self.kernels[i](x)
            # print(type(self.kernels[i]))    # <class 'torch.nn.modules.conv.Conv2d'>
            # print(x.shape)                  # torch.Size([32, 16, 48, 9])         torch.Size([32, 32, 48, 9])
            # print(tmp.shape)                # torch.Size([32, 32, 48, 9])         torch.Size([32, 16, 48, 9])
            res_list.append(tmp)
        res = torch.stack(res_list, dim=-1)
        # print(res.shape)                    # torch.Size([32, 32, 48, 9, 6])      torch.Size([32, 16, 48, 9, 6])
        res = res.mean(-1)
        # print(res.shape)                    # torch.Size([32, 32, 48, 9])         torch.Size([32, 16, 48, 9])

        # print('############# Inception_Block_V1-2')
        return res


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # print('############# TimesBlock-1')
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # print(x.shape)                  # torch.Size([32, 432, 16])
        # print(self.k)                   # 5
        # print(period_list.shape)        # (5,)
        # print(period_weight.shape)      # torch.Size([32, 5])

        res = []
        for i in range(self.k):
            # print(i)                    # 0                             1                               2                               3                               4
            period = period_list[i]
            # print(period)               # 9                             2                               2                               6                               2

            # 1. Padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # print(x.shape)              # torch.Size([32, 432, 16])
            # print(out.shape)            # torch.Size([32, 432, 16])

            # 2. Reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # print(out.shape)            # torch.Size([32, 16, 48, 9])   torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 72, 6])     torch.Size([32, 16, 216, 2])

            # 3. 2D_conv_based From 1d_tensor to 2d_tensor
            # print(type(self.conv))      # <class 'torch.nn.modules.container.Sequential'>
            # print(out.shape)            # torch.Size([32, 16, 48, 9])   torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 72, 6])     torch.Size([32, 16, 216, 2])
            out = self.conv(out)
            # print(out.shape)            # torch.Size([32, 16, 48, 9])   torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 72, 6])     torch.Size([32, 16, 216, 2])
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B, -1, N).contiguous()
            # print(out.shape)            # torch.Size([32, 432, 16])
            out = out[:, :(self.seq_len + self.pred_len), :]
            # print(out.shape)            # torch.Size([32, 432, 16])
            res.append(out)

        res = torch.stack(res, dim=-1)
        # print(res.shape)                # torch.Size([32, 432, 16, 5])

        # 4. Adaptive Aggregation
        # print(period_weight.shape)      # torch.Size([32, 5])
        period_weight = F.softmax(period_weight, dim=1)
        # print(period_weight.shape)      # torch.Size([32, 5])
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        # print(period_weight.shape)      # torch.Size([32, 432, 16, 5])
        res = torch.sum(res * period_weight, -1)
        # print(res.shape)                # torch.Size([32, 432, 16])

        res = res + x
        # print('############# TimesBlock-2')
        return res


class ModelAdapter(Model):
    def __init__(
            self,
            configs,                                                        # Backbone独有
            cffn_ratio=2.0, init_values=0., dim_ratio=2.0, period_num=3,    # Adapter独有
    ):
        super().__init__(configs=configs)
        # print('############# TimesNet_Adapter-1')
        # ######################## Original Module

        # print(self.task_name)               # long_term_forecast
        # print(self.seq_len)                 # 336
        # print(self.pred_len)                # 96
        # print(self.layer)                   # 12

        # 0. Patch Embedding (Including Position Embedding)
        # print(type(self.enc_embedding))     # <class 'layers.Embed.DataEmbedding'>
        # print(type(self.predict_linear))    # <class 'torch.nn.modules.linear.Linear'>

        # 1. Encoder Block
        # print(len(self.model))              # 12
        # print(type(self.model[0]))          # <class 'models.TimesNet.TimesBlock'>
        # print(type(self.layer_norm))        # <class 'torch.nn.modules.normalization.LayerNorm'>

        # 2. Prediction Head
        # print(type(self.projection))        # <class 'torch.nn.modules.linear.Linear'>

        # ######################## Plugin Module

        # 0. 预训练Backbone的模型参数
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.drop = configs.dropout
        # print(self.d_model)                 # 16
        # print(self.enc_in)                  # 7
        # print(self.drop)                    # 0.1

        self.normalize_layer = Normalize(self.enc_in, affine=True, non_norm=False)

        self.adapter_layer_num = period_num+1
        self.backbone_layer_num = self.layer
        assert self.backbone_layer_num >= self.adapter_layer_num, f"The Layer Num of Backbone ({self.backbone_layer_num}) is less than Adapter ({self.adapter_layer_num})"
        split_index_list = split_integer(self.backbone_layer_num, self.adapter_layer_num)
        self.interaction_indexes = [[sum(split_index_list[:i]), sum(split_index_list[:i+1])-1] for i in range(self.adapter_layer_num)]
        # print(self.backbone_layer_num)      # 12
        # print(self.adapter_layer_num)       # 4
        # print(split_index_list)             # [3, 3, 3, 3]
        # print(self.interaction_indexes)     # [[0, 2], [3, 5], [6, 8], [9, 11]]

        self.d_model_1 = int(configs.d_model // 2)
        self.d_model_2 = configs.d_model
        self.d_model_3 = int(configs.d_model * 2)
        # print(self.d_model_1)               # 8
        # print(self.d_model_2)               # 16
        # print(self.d_model_3)               # 32

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.hidden_len = self.seq_len + self.pred_len
            # print(self.hidden_len)          # 432
        else:
            self.hidden_len = self.seq_len
            # print(self.hidden_len)          # 336

        # 1. SPM, 用于生成multi-scale的c
        self.spm = TMPTemporalEmbedding(
            seq_len=self.seq_len, hidden_len=self.hidden_len, enc_in=self.enc_in,
            d_model_1=self.d_model_1, d_model_2=self.d_model_2, d_model_3=self.d_model_3,
            embed_type=configs.embed, freq=configs.freq, dropout=self.drop
        )
        self.spm.apply(self._init_weights)
        # print(type(self.spm))               # <class 'adapter_modules.comer_modules.TMPTemporalEmbedding'>

        # 2. Multi-Level Embedding, 用于为multi-scale的c添加层级嵌入信息
        self.level_embed = nn.Parameter(torch.zeros(period_num, self.hidden_len))
        normal_(self.level_embed)
        # print(self.level_embed.shape)       # torch.Size([3, 432])

        # 3. 基于BackboneBlock封装得到的AdapterBlock, 其中负责将c和x融合，并
        self.interactions = nn.Sequential(*[
            AdapterTemporalBlock(
                hidden_len=self.hidden_len, cffn_ratio=cffn_ratio, drop=self.drop,
                init_values=init_values, dim_ratio=dim_ratio, extra_CTI=(True if i == len(self.interaction_indexes) - 1 else False)
            )
            for i in range(len(self.interaction_indexes))
        ])
        self.interactions.apply(self._init_weights)

        # 4. Decode Head
        self.norm1 = nn.LayerNorm(self.d_model_1)
        self.norm2 = nn.LayerNorm(self.d_model_2)
        self.norm3 = nn.LayerNorm(self.d_model_3)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head_ppm = DecodeHeadTemporal(enc_in=self.enc_in, token_num_max=self.d_model_3, hidden_len=self.hidden_len)
            self.up_xt = nn.Conv1d(in_channels=self.d_model_2, out_channels=self.enc_in, kernel_size=(1,))
            self.up_xt.apply(self._init_weights)
            self.up_ct = nn.Conv1d(in_channels=self.seq_len, out_channels=self.hidden_len, kernel_size=(1,))
            self.up_ct.apply(self._init_weights)

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head_ppm = DecodeHeadTemporal(enc_in=self.enc_in, token_num_max=self.d_model_3, hidden_len=self.hidden_len)
            self.up_xt = nn.Conv1d(in_channels=self.d_model_2, out_channels=self.enc_in, kernel_size=(1,))
            self.up_xt.apply(self._init_weights)
            self.up_ct = nn.Conv1d(in_channels=self.seq_len, out_channels=self.hidden_len, kernel_size=(1,))
            self.up_ct.apply(self._init_weights)

        elif self.task_name == 'classification':
            self.normt = nn.LayerNorm(self.d_model)
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                in_features=(self.d_model_1 + self.d_model_2*2 + self.d_model_3)*self.hidden_len,
                out_features=configs.num_class
            )

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
        # print('############# TimesNet_Adapter.forecast-1')
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x_enc = self.normalize_layer(x_enc, 'norm')
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])

        # 1. SPM forward
        # SPM即Spatial Pyramid Matching, 是一种利用空间金字塔进行图像匹配、识别、分类的算法, SPM在不同分辨率上统计图像特征点分布，从而获取图像的局部信息。
        # print(type(self.spm))               # <class 'adapter_modules.comer_modules.TMPTemporalEmbedding'>
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        # print(x_mark_enc)                   # None
        ct, c1, c2, c3 = self.spm(x_enc, x_mark_enc)
        # print(ct.shape)                     # torch.Size([32, 336, 7])
        # print(c1.shape)                     # torch.Size([32, 432, 8])
        # print(c2.shape)                     # torch.Size([32, 432, 16])
        # print(c3.shape)                     # torch.Size([32, 432, 32])

        # 2. Multi-Level Embedding
        # print(self.level_embed[0].shape)    # torch.Size([432])
        # print(self.level_embed[1].shape)    # torch.Size([432])
        # print(self.level_embed[2].shape)    # torch.Size([432])
        c1 = c1 + self.level_embed[0].unsqueeze(0).unsqueeze(2)
        c2 = c2 + self.level_embed[1].unsqueeze(0).unsqueeze(2)
        c3 = c3 + self.level_embed[2].unsqueeze(0).unsqueeze(2)
        # print(c1.shape)                     # torch.Size([32, 432, 8])
        # print(c2.shape)                     # torch.Size([32, 432, 16])
        # print(c3.shape)                     # torch.Size([32, 432, 32])
        c = torch.cat([c1, c2, c3], dim=2)
        # print(c.shape)                      # torch.Size([32, 432, 56])

        # 3. Temporal & Position Embedding
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        # print(x_mark_enc)                   # None
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print(type(self.enc_embedding))     # <class 'layers.Embed.DataEmbedding'>
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        # print(x_mark_enc)                   # None
        # print(enc_out.shape)                # torch.Size([32, 336, 16])
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # print(type(self.predict_linear))    # <class 'torch.nn.modules.linear.Linear'>
        # print(enc_out.shape)                # torch.Size([32, 432, 16])

        # 4. Backbone & Adapter
        outs = []
        # print(len(self.interactions))       # 4
        # print(len(self.model))              # 12
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            # print(indexes)                  # [0, 2]            [3, 5]              [6, 8]                  [9, 11]
            adapter_block = self.interactions[i]
            backbone_blocks = self.model[indexes[0]:indexes[-1] + 1]
            # print(type(adapter_block))      # <class 'adapter_modules.comer_modules.AdapterTemporalBlock'>
            # print(len(backbone_blocks))     # 3
            # print(type(backbone_blocks[0])) # <class 'models.TimesNet.TimesBlock'>
            # print(enc_out.shape)            # torch.Size([32, 432, 16])
            # print(c.shape)                  # torch.Size([32, 432, 56])
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, norm=self.layer_norm, idxs=[c1.shape[2], c1.shape[2]+c2.shape[2]])
            # print(enc_out.shape)            # torch.Size([32, 432, 16])
            # print(c.shape)                  # torch.Size([32, 432, 56])
            outs.append(enc_out)

        # 5.1 Split Multi-period condition (MP-cond)
        # print(c.shape)                      # torch.Size([32, 432, 56])
        c1 = c[:, :, 0:c1.shape[2]]
        c2 = c[:, :, c1.shape[2]:c1.shape[2] + c2.shape[2]]
        c3 = c[:, :, c1.shape[2] + c2.shape[2]:]
        # print(c1.shape)                     # torch.Size([32, 432, 8])
        # print(c2.shape)                     # torch.Size([32, 432, 16])
        # print(c3.shape)                     # torch.Size([32, 432, 32])

        # 5.2 Fusion Multi-scale hidden feature from Adapter to Trend and Cond1,2,3
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([32, 432, 16])
        # print(x1.shape)                     # torch.Size([32, 432, 16])
        # print(x2.shape)                     # torch.Size([32, 432, 16])
        # print(x3.shape)                     # torch.Size([32, 432, 16])
        x1 = F.interpolate(x1, scale_factor=c1.shape[2] / x1.shape[2], mode='linear',align_corners=False, recompute_scale_factor=True)
        x3 = F.interpolate(x3, scale_factor=c3.shape[2] / x3.shape[2], mode='linear',align_corners=False, recompute_scale_factor=True)
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([32, 432, 8])
        # print(f2.shape)                     # torch.Size([32, 432, 16])
        # print(f3.shape)                     # torch.Size([32, 432, 32])

        # 6.1 Down-to-Up Path Decoder
        dec_out = self.head_ppm([f1, f2, f3])
        # print(type(self.head_ppm))          # <class 'Adapter4TS_1D.adapter_modules.comer_modules.DecodeHead'>
        # print(dec_out.shape)                # torch.Size([32, 432, 7])

        # 6.2 TrendResid AutoRegression
        # print(ct.shape)                     # torch.Size([32, 336, 7])
        ct = self.up_ct(ct)
        # print(ct.shape)                     # torch.Size([32, 432, 7])

        # 6.3 Fusion
        # print(xt.shape)                     # torch.Size([32, 432, 16])
        xt = self.up_xt(xt.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # print(xt.shape)                     # torch.Size([32, 432, 7])
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([32, 432, 7])

        # print(dec_out.shape)                # torch.Size([32, 432, 7])
        dec_out = self.normalize_layer(dec_out, 'denorm')
        # print(dec_out.shape)                # torch.Size([32, 432, 7])
        # print('############# TimesNet_Adapter.forecast-2')
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

        # print('############# TimesNet_Adapter.imputation-1')
        # 使用avgImputation可以普遍改善Timesformer在所有数据集上的性能
        x_enc = avgImputation(x_enc, mask)

        # 1. SPM Forward and Multi-Level Embedding
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        # print(x_mark_enc)                   # None
        ct, c1, c2, c3 = self.spm(x_enc, x_mark_enc)
        c1 = c1 + self.level_embed[0].unsqueeze(0).unsqueeze(2)
        c2 = c2 + self.level_embed[1].unsqueeze(0).unsqueeze(2)
        c3 = c3 + self.level_embed[2].unsqueeze(0).unsqueeze(2)
        c = torch.cat([c1, c2, c3], dim=2)
        # print(ct.shape)                     # torch.Size([32, 336, 7])
        # print(c1.shape)                     # torch.Size([32, 336, 8])
        # print(c2.shape)                     # torch.Size([32, 336, 16])
        # print(c3.shape)                     # torch.Size([32, 336, 32])
        # print(c.shape)                      # torch.Size([32, 336, 56])

        # 2. Temporal & Position Embedding
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        # print(x_mark_enc)                   # None
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print(enc_out.shape)                # torch.Size([32, 336, 16])

        # 3. TimesNet Backbone
        outs = []
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.model[indexes[0]:indexes[-1] + 1]
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, norm=self.layer_norm, idxs=[c1.shape[2], c1.shape[2]+c2.shape[2]])
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([32, 336, 16])
        # print(x1.shape)                     # torch.Size([32, 336, 16])
        # print(x2.shape)                     # torch.Size([32, 336, 16])
        # print(x3.shape)                     # torch.Size([32, 336, 16])

        # 4. Multi-scale Condition
        c1 = c[:, :, 0:c1.shape[2]]
        c2 = c[:, :, c1.shape[2]:c1.shape[2] + c2.shape[2]]
        c3 = c[:, :, c1.shape[2] + c2.shape[2]:]
        # print(c1.shape)                     # torch.Size([32, 336, 8])
        # print(c2.shape)                     # torch.Size([32, 336, 16])
        # print(c3.shape)                     # torch.Size([32, 336, 32])

        x1 = F.interpolate(x1, scale_factor=c1.shape[2] / x1.shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
        x3 = F.interpolate(x3, scale_factor=c3.shape[2] / x3.shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([32, 336, 8])
        # print(f2.shape)                     # torch.Size([32, 336, 16])
        # print(f3.shape)                     # torch.Size([32, 336, 32])

        # 5. Down-to-Up Path Decoder
        dec_out = self.head_ppm([f1, f2, f3])
        ct = self.up_ct(ct)
        xt = self.up_xt(xt.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([32, 336, 7])
        # print('############# TimesNet_Adapter.imputation-2')
        return dec_out

    def anomaly_detection(self, x_enc):
        # print('############# TimesNet_Adapter.anomaly_detection-1')
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 1. SPM Forward and Multi-Level Embedding
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])      torch.Size([32, 100, 55])
        ct, c1, c2, c3 = self.spm(x_enc, None)
        c1 = c1 + self.level_embed[0].unsqueeze(0).unsqueeze(2)
        c2 = c2 + self.level_embed[1].unsqueeze(0).unsqueeze(2)
        c3 = c3 + self.level_embed[2].unsqueeze(0).unsqueeze(2)
        c = torch.cat([c1, c2, c3], dim=2)
        # print(ct.shape)                     # torch.Size([32, 336, 7])      torch.Size([32, 100, 55])
        # print(c1.shape)                     # torch.Size([32, 336, 8])      torch.Size([32, 100, 8])
        # print(c2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])
        # print(c3.shape)                     # torch.Size([32, 336, 32])     torch.Size([32, 100, 32])
        # print(c.shape)                      # torch.Size([32, 336, 56])     torch.Size([32, 100, 56])

        # 2. Temporal & Position Embedding
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])      torch.Size([32, 100, 55])
        enc_out = self.enc_embedding(x_enc, None)
        # print(enc_out.shape)                # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])

        # 3. TimesNet Backbone
        outs = []
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.model[indexes[0]:indexes[-1] + 1]
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, norm=self.layer_norm, idxs=[c1.shape[2], c1.shape[2] + c2.shape[2]])
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])
        # print(x1.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])
        # print(x2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])
        # print(x3.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])

        # 4. Multi-scale Condition
        c1 = c[:, :, 0:c1.shape[2]]
        c2 = c[:, :, c1.shape[2]:c1.shape[2] + c2.shape[2]]
        c3 = c[:, :, c1.shape[2] + c2.shape[2]:]
        # print(c1.shape)                     # torch.Size([32, 336, 8])      torch.Size([32, 100, 8])
        # print(c2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])
        # print(c3.shape)                     # torch.Size([32, 336, 32])     torch.Size([32, 100, 32])

        x1 = F.interpolate(x1, scale_factor=c1.shape[2] / x1.shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
        x3 = F.interpolate(x3, scale_factor=c3.shape[2] / x3.shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([32, 336, 8])      torch.Size([32, 100, 8])
        # print(f2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 100, 16])
        # print(f3.shape)                     # torch.Size([32, 336, 32])     torch.Size([32, 100, 32])

        # 5. Down-to-Up Path Decoder
        dec_out = self.head_ppm([f1, f2, f3])
        ct = self.up_ct(ct)
        xt = self.up_xt(xt.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([32, 336, 7])      torch.Size([32, 100, 55])

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        # print('############# TimesNet_Adapter.anomaly_detection-2')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # print('############# TimesNet_Adapter.classification-1')

        # 1. SPM Forward and Multi-Level Embedding
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])      torch.Size([32, 1751, 3])
        ct, c1, c2, c3 = self.spm(x_enc, None)
        c1 = c1 + self.level_embed[0].unsqueeze(0).unsqueeze(2)
        c2 = c2 + self.level_embed[1].unsqueeze(0).unsqueeze(2)
        c3 = c3 + self.level_embed[2].unsqueeze(0).unsqueeze(2)
        c = torch.cat([c1, c2, c3], dim=2)
        # print(ct.shape)                     # torch.Size([32, 336, 7])      torch.Size([32, 1751, 3])
        # print(c1.shape)                     # torch.Size([32, 336, 8])      torch.Size([32, 1751, 8])
        # print(c2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        # print(c3.shape)                     # torch.Size([32, 336, 32])     torch.Size([32, 1751, 32])
        # print(c.shape)                      # torch.Size([32, 336, 56])     torch.Size([32, 1751, 56])

        # 2. Temporal & Position Embedding
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])      torch.Size([32, 1751, 3])
        enc_out = self.enc_embedding(x_enc, None)
        # print(enc_out.shape)                # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])

        # 3. TimesNet Backbone
        outs = []
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.model[indexes[0]:indexes[-1] + 1]
            enc_out, c = adapter_block(enc_out, c, backbone_blocks, norm=self.layer_norm, idxs=[c1.shape[2], c1.shape[2] + c2.shape[2]])
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        # print(x1.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        # print(x2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        # print(x3.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])

        # 4. Multi-scale Condition
        c1 = c[:, :, 0:c1.shape[2]]
        c2 = c[:, :, c1.shape[2]:c1.shape[2] + c2.shape[2]]
        c3 = c[:, :, c1.shape[2] + c2.shape[2]:]
        # print(c1.shape)                     # torch.Size([32, 336, 8])      torch.Size([32, 1751, 8])
        # print(c2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        # print(c3.shape)                     # torch.Size([32, 336, 32])     torch.Size([32, 1751, 32])

        x1 = F.interpolate(x1, scale_factor=c1.shape[2] / x1.shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
        x3 = F.interpolate(x3, scale_factor=c3.shape[2] / x3.shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([32, 336, 8])      torch.Size([32, 1751, 8])
        # print(f2.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        # print(f3.shape)                     # torch.Size([32, 336, 32])     torch.Size([32, 1751, 32])

        # Concatenate
        ct = self.enc_embedding(ct, None)
        # print(ct.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        ft = self.normt(ct + xt)
        # print(ft.shape)                     # torch.Size([32, 336, 16])     torch.Size([32, 1751, 16])
        output = torch.concat([ft, f1, f2, f3], dim=2)
        # print(output.shape)                 # torch.Size([32, 336, 88])     torch.Size([32, 1751, 75])

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(output)
        output = self.dropout(output)
        # zero-out padding embeddings
        # print(x_mark_enc.shape)             # torch.Size([32, 336])         torch.Size([32, 1751])
        output = output * x_mark_enc.unsqueeze(-1)
        # print(output.shape)                 # torch.Size([32, 336, 88])     torch.Size([32, 1751, 75])

        # Classifier
        output = output.reshape(output.shape[0], -1).contiguous()
        # print(output.shape)                 # torch.Size([32, 29568])       torch.Size([32, 131325])
        output = self.projection(output)
        # print(output.shape)                 # torch.Size([32, 4])           torch.Size([32, 4])

        # print('############# TimesNet_Adapter.classification-2')
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
