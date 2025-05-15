import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter_modules.comer_modules import MRFP, Extractor_CTI, CTI_toC, CTI_toV, PPM

from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ


# TPM_TimeEmbedding
from layers.Embed import DataEmbedding


class AdapterTemporalBlock(nn.Module):
    def __init__(self, hidden_len=432, num_heads=16, cffn_ratio=2.0, drop=0.1, init_values=0., extra_CTI=False, dim_ratio=2.0):
        super().__init__()

        self.cti_tov = CTI_toV(dim=hidden_len, num_heads=num_heads, init_values=init_values, drop=drop, cffn_ratio=cffn_ratio)
        self.cti_toc = CTI_toC(dim=hidden_len, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
        if extra_CTI:
            self.extra_CTIs = Extractor_CTI(dim=hidden_len, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
        else:
            self.extra_CTIs = None

        self.mrfp = MRFP(in_features=hidden_len, hidden_features=int(hidden_len * dim_ratio))

    def forward(self, x, c, backbone_blocks, norm, idxs):
        # print('############# AdapterTemporalBlock-1')
        # print(x.shape)                      # torch.Size([32, 432, 16])
        # print(c.shape)                      # torch.Size([32, 432, 56])
        x = x.permute(0, 2, 1).contiguous()
        c = c.permute(0, 2, 1).contiguous()
        # print(x.shape)                      # torch.Size([32, 16, 432])
        # print(c.shape)                      # torch.Size([32, 56, 432])

        c = self.mrfp(c, idxs)
        # print(c.shape)                      # torch.Size([32, 56, 432])

        c1 = c[:, 0:idxs[0], :].contiguous()
        c2 = c[:, idxs[0]:idxs[1], :].contiguous()
        c3 = c[:, idxs[1]:, :].contiguous()
        # print(c1.shape)                     # torch.Size([32, 8, 432])
        # print(c2.shape)                     # torch.Size([32, 16, 423])
        # print(c3.shape)                     # torch.Size([32, 32, 423])
        # print(x.shape)                      # torch.Size([32, 16, 432])
        c2 = c2 + x

        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)                      # torch.Size([32, 56, 432])

        # print(type(self.cti_tov))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.CTI_toV'>
        # print(x.shape)                      # torch.Size([32, 16, 432])
        # print(c.shape)                      # torch.Size([32, 56, 432])
        x = self.cti_tov(x=x, c=c, idxs=idxs)
        # print(x.shape)                      # torch.Size([32, 16, 432])

        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)                      # torch.Size([32, 432, 16])
        # print(len(backbone_blocks))         # 3
        for backbone_block in backbone_blocks:
            # print(type(backbone_block))     # <class 'models.TimesNet.TimesBlock'>
            # print(type(norm))               # <class 'torch.nn.modules.normalization.LayerNorm'>
            # print(x.shape)                  # torch.Size([32, 432, 16])
            x = norm(backbone_block(x))
            # print(x.shape)                  # torch.Size([32, 432, 16])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)                      # torch.Size([32, 16, 432])

        # print(type(self.cti_toc))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.CTI_toC'>
        # print(x.shape)                      # torch.Size([32, 16, 432])
        # print(c.shape)                      # torch.Size([32, 56, 432])
        c = self.cti_toc(x=x, c=c, idxs=idxs)
        # print(c.shape)                      # torch.Size([32, 56, 432])

        # print(type(self.extra_CTIs))        # None    None    None    <class 'Adapter4TS_1D.adapter_modules.comer_modules.Extractor_CTI'>
        if self.extra_CTIs is not None:
            # print(x.shape)                  # torch.Size([32, 16, 432])
            # print(c.shape)                  # torch.Size([32, 56, 432])
            c = self.extra_CTIs(x=x, c=c, idxs=idxs)
            # print(c.shape)                  # torch.Size([32, 56, 432])

        # print(x.shape)                      # torch.Size([32, 16, 432])
        # print(c.shape)                      # torch.Size([32, 56, 432])
        x = x.permute(0, 2, 1).contiguous()
        c = c.permute(0, 2, 1).contiguous()
        # print(x.shape)                      # torch.Size([32, 432, 16])
        # print(c.shape)                      # torch.Size([32, 432, 56])
        # print('############# AdapterTemporalBlock-2')
        return x, c


# 输入: x(32,336,7)
# 输出: ct(32,336,7)    c1(32,432,8)    c2(32,432,16)    c3(32,432,32)    cr(32,336,7)
class TMPTemporalEmbedding_v1_pre(nn.Module):
    def __init__(self, seq_len, hidden_len, enc_in, d_model_1, d_model_2, d_model_3, embed_type, freq, dropout):
        super().__init__()

        self.enc_embedding_1 = DataEmbedding(enc_in, d_model_1, embed_type, freq, dropout)
        self.enc_embedding_2 = DataEmbedding(enc_in, d_model_2, embed_type, freq, dropout)
        self.enc_embedding_3 = DataEmbedding(enc_in, d_model_3, embed_type, freq, dropout)

        self.predict_linear_1 = nn.Linear(seq_len, hidden_len)
        self.predict_linear_2 = nn.Linear(seq_len, hidden_len)
        self.predict_linear_3 = nn.Linear(seq_len, hidden_len)

    def forward(self, x, x_mark_enc):
        # print('############# TMPTemporalEmbedding-1')

        # print(x.shape)              # torch.Size([32, 336, 7])
        from adapter_modules.trend_multi_period_quantized_pool import TMPQ
        tmpq_dict = TMPQ(x, period_num_choose_first=8, period_num_choose_last=3, period_num_max=3)
        ct = tmpq_dict['trend']
        c1 = tmpq_dict['seasonal1']
        c2 = tmpq_dict['seasonal2']
        c3 = tmpq_dict['seasonal3']
        cr = tmpq_dict['resid']
        # print(ct.shape)             # torch.Size([32, 336, 7])
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(c3.shape)             # torch.Size([32, 336, 7])
        # print(cr.shape)             # torch.Size([32, 336, 7])

        # print(x_mark_enc)           # None
        c1 = self.enc_embedding_1(c1, x_mark_enc)
        c2 = self.enc_embedding_2(c2, x_mark_enc)
        c3 = self.enc_embedding_3(c3, x_mark_enc)
        # print(c1.shape)             # torch.Size([32, 336, 8])
        # print(c2.shape)             # torch.Size([32, 336, 16])
        # print(c3.shape)             # torch.Size([32, 336, 32])
        c1 = self.predict_linear_1(c1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        c2 = self.predict_linear_2(c2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        c3 = self.predict_linear_3(c3.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # print(c1.shape)             # torch.Size([32, 432, 8])
        # print(c2.shape)             # torch.Size([32, 432, 16])
        # print(c3.shape)             # torch.Size([32, 432, 32])

        # print('############# TMPTemporalEmbedding-2')
        return ct, c1, c2, c3, cr


# 输入: x(32,336,7)
# 输出: ct(32,336,7)    c1(32,432,8)    c2(32,432,16)    c3(32,432,32)    cr(32,336,7)
class TMPTemporalEmbedding_v1(nn.Module):
    def __init__(self, seq_len, hidden_len, enc_in, d_model_1, d_model_2, d_model_3, embed_type, freq, dropout):
        super().__init__()

        self.enc_embedding_1 = DataEmbedding(enc_in, d_model_1, embed_type, freq, dropout)
        self.enc_embedding_2 = DataEmbedding(enc_in, d_model_2, embed_type, freq, dropout)
        self.enc_embedding_3 = DataEmbedding(enc_in, d_model_3, embed_type, freq, dropout)

        self.predict_linear_1 = nn.Linear(seq_len, hidden_len)
        self.predict_linear_2 = nn.Linear(seq_len, hidden_len)
        self.predict_linear_3 = nn.Linear(seq_len, hidden_len)

    def forward(self, c_enc, x_mark_enc):
        # print('############# TMPTemporalEmbedding-1')

        # print(c_enc.shape)          # torch.Size([32, 7, 5, 336])
        c_enc = c_enc.permute(0, 3, 2, 1).contiguous()
        # print(c_enc.shape)          # torch.Size([32, 336, 5, 7])
        ct, c1, c2, c3, cr = c_enc[:, :, 0, :], c_enc[:, :, 1, :], c_enc[:, :, 2, :], c_enc[:, :, 3, :], c_enc[:, :, 4,
                                                                                                         :]
        # print(ct.shape)             # torch.Size([32, 336, 7])
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(c3.shape)             # torch.Size([32, 336, 7])
        # print(cr.shape)             # torch.Size([32, 336, 7])

        # print(x_mark_enc)           # None
        c1 = self.enc_embedding_1(c1, x_mark_enc)
        c2 = self.enc_embedding_2(c2, x_mark_enc)
        c3 = self.enc_embedding_3(c3, x_mark_enc)
        # print(c1.shape)             # torch.Size([32, 336, 8])
        # print(c2.shape)             # torch.Size([32, 336, 16])
        # print(c3.shape)             # torch.Size([32, 336, 32])
        c1 = self.predict_linear_1(c1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        c2 = self.predict_linear_2(c2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        c3 = self.predict_linear_3(c3.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # print(c1.shape)             # torch.Size([32, 432, 8])
        # print(c2.shape)             # torch.Size([32, 432, 16])
        # print(c3.shape)             # torch.Size([32, 432, 32])

        # print('############# TMPTemporalEmbedding-2')
        return ct, c1, c2, c3, cr


class TMPTemporalEmbedding(nn.Module):
    def __init__(self, seq_len, hidden_len, enc_in, d_model_1, d_model_2, d_model_3, embed_type, freq, dropout):
        super().__init__()

        self.enc_embedding_1 = DataEmbedding(enc_in, d_model_1, embed_type, freq, dropout)
        self.enc_embedding_2 = DataEmbedding(enc_in, d_model_2, embed_type, freq, dropout)
        self.enc_embedding_3 = DataEmbedding(enc_in, d_model_3, embed_type, freq, dropout)

        self.predict_linear_1 = nn.Linear(seq_len, hidden_len)
        self.predict_linear_2 = nn.Linear(seq_len, hidden_len)
        self.predict_linear_3 = nn.Linear(seq_len, hidden_len)

    def forward(self, x, x_mark_enc):
        # print('############# TMPTemporalEmbedding-1')
        # print(x.shape)              # torch.Size([32, 336, 7])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)              # torch.Size([32, 7, 336])

        tmpq_dict = TMPQ(x)
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        # print(ct.shape)             # torch.Size([32, 336, 7])
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(c3.shape)             # torch.Size([32, 336, 7])

        # print(x_mark_enc)           # None
        c1 = self.enc_embedding_1(c1, x_mark_enc)
        c2 = self.enc_embedding_2(c2, x_mark_enc)
        c3 = self.enc_embedding_3(c3, x_mark_enc)
        # print(c1.shape)             # torch.Size([32, 336, 8])
        # print(c2.shape)             # torch.Size([32, 336, 16])
        # print(c3.shape)             # torch.Size([32, 336, 32])
        c1 = self.predict_linear_1(c1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        c2 = self.predict_linear_2(c2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        c3 = self.predict_linear_3(c3.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # print(c1.shape)             # torch.Size([32, 432, 8])
        # print(c2.shape)             # torch.Size([32, 432, 16])
        # print(c3.shape)             # torch.Size([32, 432, 32])

        # print('############# TMPTemporalEmbedding-2')
        return ct, c1, c2, c3


# 输入: f1(32,432,8), f2(32,432,16), f3(32,432,32)
# 输出: dec_out(32,432,7)
class DecodeHeadTemporal(nn.Module):
    def __init__(self, enc_in, token_num_max, hidden_len, pool_scales=(1, 2, 3, 6), period_num=3,):
        super(DecodeHeadTemporal, self).__init__()
        self.enc_in = enc_in
        self.hidden_len = hidden_len
        self.period_num = period_num
        self.token_num_max = token_num_max

        # PSP Module
        self.psp_modules = PPM(pool_scales=pool_scales, in_channel=self.hidden_len, out_channel=self.hidden_len)
        self.bottleneck = nn.Conv1d(in_channels=(len(pool_scales)+1) * self.hidden_len, out_channels=self.hidden_len, kernel_size=(3,), padding=1)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for period_idx in range(period_num-1):
            l_conv = nn.Conv1d(in_channels=self.hidden_len, out_channels=self.hidden_len, kernel_size=(1,))
            fpn_conv = nn.Conv1d(in_channels=self.hidden_len, out_channels=self.hidden_len, kernel_size=(3,), padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # FPM Bottleneck
        self.fpn_bottleneck = nn.Conv1d(in_channels=int(self.period_num*self.hidden_len), out_channels=self.hidden_len, kernel_size=(3,), padding=1)
        self.conv_seg = nn.Conv1d(in_channels=self.hidden_len, out_channels=self.hidden_len, kernel_size=(1,))

        # Output
        self.output = nn.Conv1d(in_channels=self.token_num_max, out_channels=self.enc_in, kernel_size=(1,))

        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        # print('############# decode_head-1')

        # 1. Conv-based Build Lateral.1-2
        # print(len(self.lateral_convs))          # 2
        # print(len(inputs))                      # 3
        laterals = []
        for i in range(len(self.lateral_convs)):
            lateral_conv = self.lateral_convs[i]
            input = inputs[i]
            lateral = lateral_conv(input)
            # print(type(lateral_conv))           # <class 'torch.nn.modules.conv.Conv1d'>
            # print(input.shape)                  # torch.Size([32, 432, 8])     torch.Size([32, 432, 16])
            # print(lateral.shape)                # torch.Size([32, 432, 8])     torch.Size([32, 432, 16])
            laterals.append(lateral)

        # 2. PPM_based Build Lateral.3
        tmp = inputs[-1]
        # print(tmp.shape)                        # torch.Size([32, 432, 32])
        psp_outs = [tmp]
        psp_out = self.psp_modules(tmp)
        # print(type(self.psp_modules))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.PPM'>
        # print(tmp.shape)                        # torch.Size([32, 432, 32])
        # print(len(psp_out))                     # 4
        # for tmp in psp_out:
        #     print(tmp.shape)                    # (32,432,32)    (32,432,32)    (32,432,32)    (32,432,32)
        psp_outs.extend(psp_out)
        psp_outs = torch.cat(psp_outs, dim=1)
        # print(psp_outs.shape)                   # torch.Size([32, 2160, 32])

        psp_out = self.bottleneck(psp_outs)
        # print(type(self.bottleneck))            # <class 'torch.nn.modules.conv.Conv1d'>
        # print(psp_outs.shape)                   # torch.Size([32, 2160, 32])
        # print(psp_out.shape)                    # torch.Size([32, 432, 32])
        laterals.append(psp_out)

        # 3. FPN Top-Down Path, Lateral.1-3 Fusion
        # print(len(laterals))                    # 3
        # for tmp in laterals:
        #     print(tmp.shape)                    # (32,432,8)   (32,432,16)   (32,432,32)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            # print(laterals[i].shape)            # torch.Size([32, 432, 8])      torch.Size([32, 432, 16])
            lateral_tmp = F.interpolate(laterals[i], scale_factor=laterals[i+1].shape[2]/laterals[i].shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
            # print(lateral_tmp.shape)            # torch.Size([32, 432, 16])     torch.Size([32, 432, 32])
            # print(laterals[i + 1].shape)        # torch.Size([32, 432, 16])     torch.Size([32, 432, 32])
            laterals[i + 1] = laterals[i + 1] + lateral_tmp
            # print(laterals[i + 1].shape)        # torch.Size([32, 432, 16])     torch.Size([32, 432, 32])

        # 4. ShapeKeep Conv, Lateral.1-2
        # print(used_backbone_levels)             # 3
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            layer = self.fpn_convs[i]
            lateral = laterals[i]
            fpn_out = layer(lateral)
            # print(type(layer))                  # <class 'torch.nn.modules.conv.Conv1d'>
            # print(lateral.shape)                # torch.Size([32, 432, 8])      torch.Size([32, 432, 16])
            # print(fpn_out.shape)                # torch.Size([32, 432, 8])      torch.Size([32, 432, 16])
            fpn_outs.append(fpn_out)

        # print(laterals[-1].shape)               # torch.Size([32, 432, 32])
        fpn_outs.append(laterals[-1])

        # 5. Reshape: Lateral.2-3 -> Lateral.1
        # print(used_backbone_levels)             # 3
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            # print(fpn_outs[i].shape)            # torch.Size([32, 432, 8])      torch.Size([32, 432, 16])
            fpn_outs[i] = F.interpolate(fpn_outs[i], scale_factor=fpn_outs[-1].shape[2]/fpn_outs[i].shape[2], mode='linear', align_corners=False, recompute_scale_factor=True)
            # print(fpn_outs[i].shape)            # torch.Size([32, 432, 32])     torch.Size([32, 432, 32])

        # 6. Concat Lateral.1-3 DownConv Bottleneck
        # print(len(fpn_outs))                    # 3
        # for fpn_out in fpn_outs:
        #     print(fpn_out.shape)                # torch.Size([32, 432, 32])     torch.Size([32, 432, 32])       torch.Size([32, 432, 32])
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # print(fpn_outs.shape)                   # torch.Size([32, 1296, 32])

        feat = self.fpn_bottleneck(fpn_outs)
        # print(type(self.fpn_bottleneck))        # <class 'torch.nn.modules.conv.Conv1d'>
        # print(fpn_outs.shape)                   # torch.Size([32, 1296, 32])
        # print(feat.shape)                       # torch.Size([32, 432, 32])
        if self.dropout is not None:
            feat = self.dropout(feat)
        seasonal = self.conv_seg(feat)
        # print(seasonal.shape)                   # torch.Size([32, 432, 32])
        seasonal = self.output(seasonal.transpose(1, 2)).transpose(1, 2)
        # print(seasonal.shape)                   # torch.Size([32, 432, 7])

        # print('############# decode_head-2')
        return seasonal

