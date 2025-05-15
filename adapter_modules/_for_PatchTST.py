import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter_modules.comer_modules import MRFP, Extractor_CTI, CTI_toC, CTI_toV, PPM

# TPM_PatchEmbedding
from layers.Embed import PositionalEmbedding, TokenEmbedding

from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ


class AdapterPatchBlock(nn.Module):
    def __init__(self, dim=32, num_heads=4, cffn_ratio=2.0, drop=0.1, init_values=0., extra_CTI=False, dim_ratio=6.0):
        super().__init__()

        self.cti_tov = CTI_toV(dim=dim, num_heads=num_heads, init_values=init_values, drop=drop, cffn_ratio=cffn_ratio)
        self.cti_toc = CTI_toC(dim=dim, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
        if extra_CTI:
            self.extra_CTIs = Extractor_CTI(dim=dim, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
        else:
            self.extra_CTIs = None

        self.mrfp = MRFP(in_features=dim, hidden_features=int(dim * dim_ratio))

    def forward(self, x, c, backbone_blocks, idxs):
        # print('############# AdapterTemporalBlock-1')
        # print(c.shape)                      # torch.Size([224, 112, 32])

        # print(c.shape)                      # torch.Size([224, 112, 32])
        c = self.mrfp(c, idxs)
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

        # print(type(self.cti_tov))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.CTI_toV'>
        # print(x.shape)                      # torch.Size([224, 32, 32])
        # print(c.shape)                      # torch.Size([224, 112, 32])
        x = self.cti_tov(x=x, c=c, idxs=idxs)
        # print(x.shape)                      # torch.Size([224, 32, 32])

        # print(len(backbone_blocks))         # 3
        for backbone_block in backbone_blocks:
            # print(type(backbone_block))     # <class 'Adapter4TS_1D.layers.Transformer_EncDec.EncoderLayer'>
            # print(x.shape)                  # torch.Size([224, 32, 32])
            x, _ = backbone_block(x=x, attn_mask=None, tau=None, delta=None)
            # print(x.shape)                  # torch.Size([224, 32, 32])

        # print(type(self.cti_toc))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.CTI_toC'>
        # print(x.shape)                      # torch.Size([224, 32, 32])
        # print(c.shape)                      # torch.Size([224, 112, 32])
        c = self.cti_toc(x=x, c=c, idxs=idxs)
        # print(c.shape)                      # torch.Size([224, 112, 32])

        # print(type(self.extra_CTIs))        # None    None    None    <class 'Adapter4TS_1D.adapter_modules.comer_modules.Extractor_CTI'>
        if self.extra_CTIs is not None:
            # print(x.shape)                  # torch.Size([224, 32, 32])
            # print(c.shape)                  # torch.Size([224, 112, 32])
            c = self.extra_CTIs(x=x, c=c, idxs=idxs)
            # print(c.shape)                  # torch.Size([224, 112, 32])

        # print(x.shape)                      # torch.Size([224, 32, 32])
        # print(c.shape)                      # torch.Size([224, 112, 32])
        # print('############# AdapterTemporalBlock-2')
        return x, c


# Spatial Pyramid Matching (SPM)
# 输入: x(32,7,336)
# 输出: ct(224,336)    c1(224,19,32)    c2(224,40,32)    c3(224,82,32)    cr(224,336)
class TMPPatchEmbedding_v1_pre(nn.Module):
    def __init__(
            self, d_model,
            patch_len_1, patch_len_2, patch_len_3,
            stride_1, stride_2, stride_3,
            patch_num_1, patch_num_2, patch_num_3, dropout=0.1
    ):
        super().__init__()

        self.patch_len_1 = patch_len_1
        self.patch_len_2 = patch_len_2
        self.patch_len_3 = patch_len_3
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.stride_3 = stride_3

        # Patching padding
        self.padding_patch_layer_1 = nn.ReplicationPad1d((0, self.stride_1))
        self.padding_patch_layer_2 = nn.ReplicationPad1d((0, self.stride_2))
        self.padding_patch_layer_3 = nn.ReplicationPad1d((0, self.stride_3))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding_1 = TokenEmbedding(self.patch_len_1, d_model)
        self.value_embedding_2 = TokenEmbedding(self.patch_len_2, d_model)
        self.value_embedding_3 = TokenEmbedding(self.patch_len_3, d_model)

        # Channel
        self.channel_embedding_1 = nn.Conv1d(patch_num_1, patch_num_1, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_2 = nn.Conv1d(patch_num_2, patch_num_2, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_3 = nn.Conv1d(patch_num_3, patch_num_3, kernel_size=1, stride=1, padding=0, bias=True)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('############# TMPPatchEmbedding-1')

        # print(x.shape)              # torch.Size([32, 7, 336])
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])).unsqueeze(2).contiguous()
        # print(x.shape)              # torch.Size([224, 336, 1])

        """
        ct, c1, c2, c3, cr = TMPQ(x, period_num_choose_first=8, period_num_choose_last=3, period_num_max=3)
        """
        from adapter_modules.trend_multi_period_quantized_pool import TMPQ
        tmpq_dict = TMPQ(x, period_num_choose_first=8, period_num_choose_last=3, period_num_max=3)
        ct = tmpq_dict['trend'].squeeze(2)
        c1 = tmpq_dict['seasonal1'].squeeze(2)
        c2 = tmpq_dict['seasonal2'].squeeze(2)
        c3 = tmpq_dict['seasonal3'].squeeze(2)
        cr = tmpq_dict['resid'].squeeze(2)
        # print(ct.shape)             # torch.Size([224, 336])
        # print(c1.shape)             # torch.Size([224, 336])
        # print(c2.shape)             # torch.Size([224, 336])
        # print(c3.shape)             # torch.Size([224, 336])
        # print(cr.shape)             # torch.Size([224, 336])

        c1 = self.padding_patch_layer_1(c1)
        c2 = self.padding_patch_layer_2(c2)
        c3 = self.padding_patch_layer_3(c3)
        # print(c1.shape)             # torch.Size([224, 352])
        # print(c2.shape)             # torch.Size([224, 344])
        # print(c3.shape)             # torch.Size([224, 340])
        c1 = c1.unfold(dimension=-1, size=self.patch_len_1, step=self.stride_1)
        c2 = c2.unfold(dimension=-1, size=self.patch_len_2, step=self.stride_2)
        c3 = c3.unfold(dimension=-1, size=self.patch_len_3, step=self.stride_3)
        # print(c1.shape)             # torch.Size([224, 18, 64])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 16])

        ve1 = self.value_embedding_1(c1)
        ve2 = self.value_embedding_2(c2)
        ve3 = self.value_embedding_3(c3)
        # print(ve1.shape)            # torch.Size([224, 18, 32])
        # print(ve2.shape)            # torch.Size([224, 39, 32])
        # print(ve3.shape)            # torch.Size([224, 81, 32])

        pe1 = self.position_embedding(c1)
        pe2 = self.position_embedding(c2)
        pe3 = self.position_embedding(c3)
        # print(pe1.shape)            # torch.Size([1, 18, 32])
        # print(pe2.shape)            # torch.Size([1, 39, 32])
        # print(pe3.shape)            # torch.Size([1, 81, 32])

        c1 = self.dropout(ve1 + pe1)
        c2 = self.dropout(ve2 + pe2)
        c3 = self.dropout(ve3 + pe3)
        # print(c1.shape)             # torch.Size([224, 18, 32])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 32])

        c1 = self.channel_embedding_1(c1)
        c2 = self.channel_embedding_2(c2)
        c3 = self.channel_embedding_3(c3)
        # print(c1.shape)             # torch.Size([224, 18, 32])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 32])

        # print('############# TMPPatchEmbedding-2')
        return ct, c1, c2, c3, cr


class TMPPatchEmbedding_v1(nn.Module):
    def __init__(
            self, d_model,
            patch_len_1, patch_len_2, patch_len_3,
            stride_1, stride_2, stride_3,
            patch_num_1, patch_num_2, patch_num_3, dropout=0.1
    ):
        super().__init__()

        self.patch_len_1 = patch_len_1
        self.patch_len_2 = patch_len_2
        self.patch_len_3 = patch_len_3
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.stride_3 = stride_3

        # Patching padding
        self.padding_patch_layer_1 = nn.ReplicationPad1d((0, self.stride_1))
        self.padding_patch_layer_2 = nn.ReplicationPad1d((0, self.stride_2))
        self.padding_patch_layer_3 = nn.ReplicationPad1d((0, self.stride_3))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding_1 = TokenEmbedding(self.patch_len_1, d_model)
        self.value_embedding_2 = TokenEmbedding(self.patch_len_2, d_model)
        self.value_embedding_3 = TokenEmbedding(self.patch_len_3, d_model)

        # Channel
        self.channel_embedding_1 = nn.Conv1d(patch_num_1, patch_num_1, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_2 = nn.Conv1d(patch_num_2, patch_num_2, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_3 = nn.Conv1d(patch_num_3, patch_num_3, kernel_size=1, stride=1, padding=0, bias=True)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, c):
        # print('############# TMPPatchEmbedding-1')

        # print(c.shape)              # torch.Size([32, 7, 5, 336])
        c = torch.reshape(c, (c.shape[0] * c.shape[1], c.shape[2], c.shape[3])).contiguous()
        # print(c.shape)              # torch.Size([224, 5, 336])
        ct, c1, c2, c3, cr = c[:, 0, :], c[:, 1, :], c[:, 2, :], c[:, 3, :], c[:, 4, :]
        # print(ct.shape)             # torch.Size([224, 336])
        # print(c1.shape)             # torch.Size([224, 336])
        # print(c2.shape)             # torch.Size([224, 336])
        # print(c3.shape)             # torch.Size([224, 336])
        # print(cr.shape)             # torch.Size([224, 336])

        c1 = self.padding_patch_layer_1(c1)
        c2 = self.padding_patch_layer_2(c2)
        c3 = self.padding_patch_layer_3(c3)
        # print(c1.shape)             # torch.Size([224, 352])
        # print(c2.shape)             # torch.Size([224, 344])
        # print(c3.shape)             # torch.Size([224, 340])
        c1 = c1.unfold(dimension=-1, size=self.patch_len_1, step=self.stride_1)
        c2 = c2.unfold(dimension=-1, size=self.patch_len_2, step=self.stride_2)
        c3 = c3.unfold(dimension=-1, size=self.patch_len_3, step=self.stride_3)
        # print(c1.shape)             # torch.Size([224, 18, 64])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 16])

        ve1 = self.value_embedding_1(c1)
        ve2 = self.value_embedding_2(c2)
        ve3 = self.value_embedding_3(c3)
        # print(ve1.shape)            # torch.Size([224, 18, 32])
        # print(ve2.shape)            # torch.Size([224, 39, 32])
        # print(ve3.shape)            # torch.Size([224, 81, 32])

        pe1 = self.position_embedding(c1)
        pe2 = self.position_embedding(c2)
        pe3 = self.position_embedding(c3)
        # print(pe1.shape)            # torch.Size([1, 18, 32])
        # print(pe2.shape)            # torch.Size([1, 39, 32])
        # print(pe3.shape)            # torch.Size([1, 81, 32])

        c1 = self.dropout(ve1 + pe1)
        c2 = self.dropout(ve2 + pe2)
        c3 = self.dropout(ve3 + pe3)
        # print(c1.shape)             # torch.Size([224, 18, 32])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 32])

        c1 = self.channel_embedding_1(c1)
        c2 = self.channel_embedding_2(c2)
        c3 = self.channel_embedding_3(c3)
        # print(c1.shape)             # torch.Size([224, 18, 32])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 32])

        # print('############# TMPPatchEmbedding-2')
        return ct, c1, c2, c3, cr


class TMPPatchEmbedding(nn.Module):
    def __init__(
            self, d_model,
            patch_len_1, patch_len_2, patch_len_3,
            stride_1, stride_2, stride_3,
            patch_num_1, patch_num_2, patch_num_3, dropout=0.1
    ):
        super().__init__()

        self.patch_len_1 = patch_len_1
        self.patch_len_2 = patch_len_2
        self.patch_len_3 = patch_len_3
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.stride_3 = stride_3

        # Patching padding
        self.padding_patch_layer_1 = nn.ReplicationPad1d((0, self.stride_1))
        self.padding_patch_layer_2 = nn.ReplicationPad1d((0, self.stride_2))
        self.padding_patch_layer_3 = nn.ReplicationPad1d((0, self.stride_3))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding_1 = TokenEmbedding(self.patch_len_1, d_model)
        self.value_embedding_2 = TokenEmbedding(self.patch_len_2, d_model)
        self.value_embedding_3 = TokenEmbedding(self.patch_len_3, d_model)

        # Channel
        self.channel_embedding_1 = nn.Conv1d(patch_num_1, patch_num_1, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_2 = nn.Conv1d(patch_num_2, patch_num_2, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_3 = nn.Conv1d(patch_num_3, patch_num_3, kernel_size=1, stride=1, padding=0, bias=True)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('############# TMPPatchEmbedding-1')

        # print(x.shape)              # torch.Size([32, 7, 336])
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])).unsqueeze(1).contiguous()
        # print(x.shape)              # torch.Size([224, 1, 336])

        tmpq_dict = TMPQ(x)
        ct = tmpq_dict['trend'].squeeze(1)[:, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'].squeeze(1)[:, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'].squeeze(1)[:, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'].squeeze(1)[:, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([224, 336])
        # print(c1.shape)             # torch.Size([224, 336])
        # print(c2.shape)             # torch.Size([224, 336])
        # print(c3.shape)             # torch.Size([224, 336])

        c1 = self.padding_patch_layer_1(c1)
        c2 = self.padding_patch_layer_2(c2)
        c3 = self.padding_patch_layer_3(c3)
        # print(c1.shape)             # torch.Size([224, 352])
        # print(c2.shape)             # torch.Size([224, 344])
        # print(c3.shape)             # torch.Size([224, 340])
        c1 = c1.unfold(dimension=-1, size=self.patch_len_1, step=self.stride_1)
        c2 = c2.unfold(dimension=-1, size=self.patch_len_2, step=self.stride_2)
        c3 = c3.unfold(dimension=-1, size=self.patch_len_3, step=self.stride_3)
        # print(c1.shape)             # torch.Size([224, 18, 64])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 16])

        ve1 = self.value_embedding_1(c1)
        ve2 = self.value_embedding_2(c2)
        ve3 = self.value_embedding_3(c3)
        # print(ve1.shape)            # torch.Size([224, 18, 32])
        # print(ve2.shape)            # torch.Size([224, 39, 32])
        # print(ve3.shape)            # torch.Size([224, 81, 32])

        pe1 = self.position_embedding(c1)
        pe2 = self.position_embedding(c2)
        pe3 = self.position_embedding(c3)
        # print(pe1.shape)            # torch.Size([1, 18, 32])
        # print(pe2.shape)            # torch.Size([1, 39, 32])
        # print(pe3.shape)            # torch.Size([1, 81, 32])

        c1 = self.dropout(ve1 + pe1)
        c2 = self.dropout(ve2 + pe2)
        c3 = self.dropout(ve3 + pe3)
        # print(c1.shape)             # torch.Size([224, 18, 32])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 32])

        c1 = self.channel_embedding_1(c1)
        c2 = self.channel_embedding_2(c2)
        c3 = self.channel_embedding_3(c3)
        # print(c1.shape)             # torch.Size([224, 18, 32])
        # print(c2.shape)             # torch.Size([224, 39, 32])
        # print(c3.shape)             # torch.Size([224, 81, 32])

        # print('############# TMPPatchEmbedding-2')
        return ct, c1, c2, c3


# 输入: f1(224,19,32), f2(224,40,32), f3(224,82,32)
# 输出: dec_out(224,96)
class DecodeHeadPatch(nn.Module):
    def __init__(self, token_num_max, d_model, pred_len, pool_scales=(1, 2, 3, 6), period_num=3,):
        super(DecodeHeadPatch, self).__init__()
        self.d_model = d_model
        self.period_num = period_num
        self.pred_len = pred_len
        self.token_num_max = token_num_max

        # PSP Module
        self.psp_modules = PPM(pool_scales=pool_scales, in_channel=self.d_model, out_channel=self.d_model)
        self.bottleneck = nn.Conv1d(in_channels=(len(pool_scales)+1) * self.d_model, out_channels=self.d_model, kernel_size=(3,), padding=1)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for period_idx in range(period_num-1):
            l_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=(1,))
            fpn_conv = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=(3,), padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # FPM Bottleneck
        self.fpn_bottleneck = nn.Conv1d(in_channels=int(self.period_num*self.d_model), out_channels=self.d_model, kernel_size=(3,), padding=1)
        self.conv_seg = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=(1,))

        # Output
        self.flatten = nn.Flatten(start_dim=-2)
        self.output = nn.Linear(in_features=int(self.token_num_max * self.d_model), out_features=self.pred_len, bias=True)

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
            lateral = lateral_conv(input.transpose(1, 2)).transpose(1, 2)
            # print(type(lateral_conv))           # <class 'torch.nn.modules.conv.Conv1d'>
            # print(input.shape)                  # torch.Size([224, 19, 32])     torch.Size([224, 40, 32])
            # print(lateral.shape)                # torch.Size([224, 19, 32])     torch.Size([224, 40, 32])
            laterals.append(lateral)

        # 2. PPM_based Build Lateral.3
        # print(inputs[-1].shape)                 # torch.Size([224, 82, 32])
        tmp = inputs[-1].transpose(1, 2)
        # print(tmp.shape)                        # torch.Size([224, 32, 82])
        psp_outs = [tmp]
        psp_out = self.psp_modules(tmp)
        # print(type(self.psp_modules))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.PPM'>
        # print(tmp.shape)                        # torch.Size([224, 32, 82])
        # print(len(psp_out))                     # 4
        # for tmp in psp_out:
        #     print(tmp.shape)                    # (224,32,82)    (224,32,82)    (224,32,82)    (224,32,82)
        psp_outs.extend(psp_out)
        psp_outs = torch.cat(psp_outs, dim=1)
        # print(psp_outs.shape)                   # torch.Size([224, 160, 82])

        psp_out = self.bottleneck(psp_outs)
        # print(type(self.bottleneck))            # <class 'torch.nn.modules.conv.Conv1d'>
        # print(psp_outs.shape)                   # torch.Size([224, 160, 82])
        # print(psp_out.shape)                    # torch.Size([224, 32, 82])
        psp_out = psp_out.transpose(1, 2)
        # print(psp_out.shape)                    # torch.Size([224, 82, 32])
        laterals.append(psp_out)

        # 3. FPN Top-Down Path, Lateral.1-3 Fusion
        # print(len(laterals))                    # 3
        # for tmp in laterals:
        #     print(tmp.shape)                    # (224,19,32)   (224,40,32)   (224,82,32)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            # print(laterals[i].shape)            # torch.Size([224, 19, 32])     torch.Size([224, 40, 32])
            lateral_tmp = F.interpolate(laterals[i].transpose(1, 2), scale_factor=laterals[i+1].shape[1]/laterals[i].shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
            # print(lateral_tmp.shape)            # torch.Size([224, 40, 32])     torch.Size([224, 82, 32])
            # print(laterals[i + 1].shape)        # torch.Size([224, 40, 32])     torch.Size([224, 82, 32])
            laterals[i + 1] = laterals[i + 1] + lateral_tmp
            # print(laterals[i + 1].shape)        # torch.Size([224, 40, 32])     torch.Size([224, 82, 32])

        # 4. ShapeKeep Conv, Lateral.1-2
        # print(used_backbone_levels)             # 3
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            layer = self.fpn_convs[i]
            lateral = laterals[i]
            fpn_out = layer(lateral.transpose(1, 2)).transpose(1, 2)
            # print(type(layer))                  # <class 'torch.nn.modules.conv.Conv1d'>
            # print(lateral.shape)                # torch.Size([224, 19, 32])     torch.Size([224, 40, 32])
            # print(fpn_out.shape)                # torch.Size([224, 19, 32])     torch.Size([224, 40, 32])
            fpn_outs.append(fpn_out)

        # print(laterals[-1].shape)               # torch.Size([224, 82, 32])
        fpn_outs.append(laterals[-1])

        # 5. Reshape: Lateral.2-3 -> Lateral.1
        # print(used_backbone_levels)             # 3
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            # print(fpn_outs[i].shape)            # torch.Size([224, 19, 32])     torch.Size([224, 40, 32])
            fpn_outs[i] = F.interpolate(fpn_outs[i].transpose(1, 2), scale_factor=fpn_outs[-1].shape[1]/fpn_outs[i].shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
            # print(fpn_outs[i].shape)            # torch.Size([224, 82, 32])     torch.Size([224, 82, 32])

        # 6. Concat Lateral.1-3 DownConv Bottleneck
        # print(len(fpn_outs))                    # 3
        # for fpn_out in fpn_outs:
        #     print(fpn_out.shape)                # torch.Size([224, 82, 32])     torch.Size([224, 82, 32])       torch.Size([224, 82, 32])
        fpn_outs = torch.cat(fpn_outs, dim=2)
        # print(fpn_outs.shape)                   # torch.Size([224, 82, 96])

        feat = self.fpn_bottleneck(fpn_outs.transpose(1, 2)).transpose(1, 2)
        # print(type(self.fpn_bottleneck))        # <class 'torch.nn.modules.conv.Conv1d'>
        # print(fpn_outs.shape)                   # torch.Size([224, 82, 96])
        # print(feat.shape)                       # torch.Size([224, 82, 32])
        if self.dropout is not None:
            feat = self.dropout(feat)
        seasonal = self.conv_seg(feat.transpose(1, 2)).transpose(1, 2)
        # print(seasonal.shape)                   # torch.Size([224, 82, 32])
        seasonal = self.output(self.flatten(seasonal))
        # print(seasonal.shape)                   # torch.Size([224, 96])

        # print('############# decode_head-2')
        return seasonal

