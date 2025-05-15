import torch
import torch.nn as nn
from models.TimeMixer import DataEmbedding_wo_pos

import torch.nn.functional as F

from adapter_modules.comer_modules import PPM

from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ


# 输入:
# e1(896,336,16),e2(896,168,16),e3(896,84,16),e4(896,42,16)
# c1(896,336,64),c2(896,168,64),c3(896,84,64),c4(896,42,64)
class AdapterTimeMixerBlock(nn.Module):
    def __init__(self, d_model, cond_num):
        super().__init__()
        self.mrfp_layers = torch.nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=4 * d_model, out_features=4 * d_model),
                nn.GELU(),
                nn.Linear(in_features=4 * d_model, out_features=4 * d_model),
            ) for _ in range(cond_num)
        ])
        self.cti_tox_layers = torch.nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=4 * d_model, out_features=d_model),
                nn.GELU(),
                nn.Linear(in_features=d_model, out_features=d_model),
            ) for _ in range(cond_num)
        ])
        self.cti_toc_layers = torch.nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model),
                nn.GELU(),
                nn.Linear(in_features=d_model, out_features=4 * d_model),
            ) for _ in range(cond_num)
        ])

    def forward(self, x_list, c_list, backbone_blocks):
        # print('############# AdapterTemporalBlock-1')
        x1, x2, x3, x4 = x_list
        # print(x1.shape)                     # torch.Size([896, 336, 16])
        # print(x2.shape)                     # torch.Size([896, 168, 16])
        # print(x3.shape)                     # torch.Size([896, 84, 16])
        # print(x4.shape)                     # torch.Size([896, 42, 16])

        c1, c2, c3, c4 = c_list
        # print(c1.shape)                     # torch.Size([896, 336, 64])
        # print(c2.shape)                     # torch.Size([896, 168, 64])
        # print(c3.shape)                     # torch.Size([896, 84, 64])
        # print(c4.shape)                     # torch.Size([896, 42, 64])

        c1, c2, c3, c4 = self.mrfp_layers[0](c1), self.mrfp_layers[1](c2), self.mrfp_layers[2](c3), self.mrfp_layers[3](c4)
        # print(c1.shape)                     # torch.Size([896, 336, 64])
        # print(c2.shape)                     # torch.Size([896, 168, 64])
        # print(c3.shape)                     # torch.Size([896, 84, 64])
        # print(c4.shape)                     # torch.Size([896, 42, 64])

        x1_res, x2_res, x3_res, x4_res = self.cti_tox_layers[0](c1), self.cti_tox_layers[1](c2), self.cti_tox_layers[2](c3), self.cti_tox_layers[3](c4)
        # print(x1_res.shape)                 # torch.Size([896, 336, 16])
        # print(x2_res.shape)                 # torch.Size([896, 168, 16])
        # print(x3_res.shape)                 # torch.Size([896, 84, 16])
        # print(x4_res.shape)                 # torch.Size([896, 42, 16])

        x1, x2, x3, x4 = x1 + x1_res, x2 + x2_res, x3 + x3_res, x4 + x4_res
        # print(x1.shape)                     # torch.Size([896, 336, 16])
        # print(x2.shape)                     # torch.Size([896, 168, 16])
        # print(x3.shape)                     # torch.Size([896, 84, 16])
        # print(x4.shape)                     # torch.Size([896, 42, 16])
        x_list = [x1, x2, x3, x4]

        # print(len(backbone_blocks))         # 1
        # print(type(backbone_blocks[0]))     # <class 'models.TimeMixer.PastDecomposableMixing'>
        for backbone_block in backbone_blocks:
            x_list = backbone_block(x_list)

        x1, x2, x3, x4 = x_list
        # print(x1.shape)                     # torch.Size([896, 336, 16])
        # print(x2.shape)                     # torch.Size([896, 168, 16])
        # print(x3.shape)                     # torch.Size([896, 84, 16])
        # print(x4.shape)                     # torch.Size([896, 42, 16])

        c1_res, c2_res, c3_res, c4_res = self.cti_toc_layers[0](x1), self.cti_toc_layers[1](x2), self.cti_toc_layers[2](x3), self.cti_toc_layers[3](x4)
        # print(c1_res.shape)                 # torch.Size([896, 336, 64])
        # print(c2_res.shape)                 # torch.Size([896, 168, 64])
        # print(c3_res.shape)                 # torch.Size([896, 84, 64])
        # print(c4_res.shape)                 # torch.Size([896, 42, 64])

        c1, c2, c3, c4 = c1 + c1_res, c2 + c2_res, c3 + c3_res, c4 + c4_res
        # print(c1.shape)                     # torch.Size([896, 336, 64])
        # print(c2.shape)                     # torch.Size([896, 168, 64])
        # print(c3.shape)                     # torch.Size([896, 84, 64])
        # print(c4.shape)                     # torch.Size([896, 42, 64])
        c_list = [c1, c2, c3, c4]

        # print('############# AdapterTemporalBlock-2')
        return x_list, c_list


# 输入:
# x1(896,336,1), x2(896,168,1), x3(896,84,1), x4(896,42,1)
# 输出:
# c1(896,336,64),c2(896,168,64),c3(896,84,64),c4(896,42,64)
class TMPTimeMixerEmbedding(nn.Module):
    def __init__(self, d_model, embed, freq, dropout, channel_independence, enc_in):
        super().__init__()
        if channel_independence:
            self.enc_embedding_1 = DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)
            self.enc_embedding_2 = DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)
            self.enc_embedding_3 = DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)
            self.enc_embedding_4 = DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)
        else:
            self.enc_embedding_1 = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
            self.enc_embedding_2 = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
            self.enc_embedding_3 = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
            self.enc_embedding_4 = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)

    def forward(self, x_list, level_embed):
        # print('############# TMPTimeMixerEmbedding-1')

        x1, x2, x3, x4 = x_list
        x1 = x1.permute(0, 2, 1).contiguous()
        x2 = x2.permute(0, 2, 1).contiguous()
        x3 = x3.permute(0, 2, 1).contiguous()
        x4 = x4.permute(0, 2, 1).contiguous()
        # print(x1.shape)             # torch.Size([896, 1, 336])
        # print(x2.shape)             # torch.Size([896, 1, 168])
        # print(x3.shape)             # torch.Size([896, 1, 84])
        # print(x4.shape)             # torch.Size([896, 1, 42])

        tmpq_dict1 = TMPQ(x1)
        x1t = tmpq_dict1['trend'][:, :, :x1.shape[-1]].permute(0, 2, 1).contiguous()
        x11 = tmpq_dict1['seasonal_1'].permute(0, 2, 1).contiguous()
        x12 = tmpq_dict1['seasonal_2'].permute(0, 2, 1).contiguous()
        x13 = tmpq_dict1['seasonal_3'].permute(0, 2, 1).contiguous()
        # print(x1t.shape)            # torch.Size([896, 336, 1])
        # print(x11.shape)            # torch.Size([896, 336, 1])
        # print(x12.shape)            # torch.Size([896, 336, 1])
        # print(x13.shape)            # torch.Size([896, 336, 1])

        tmpq_dict2 = TMPQ(x2)
        x2t = tmpq_dict2['trend'].permute(0, 2, 1).contiguous()
        x21 = tmpq_dict2['seasonal_1'].permute(0, 2, 1).contiguous()
        x22 = tmpq_dict2['seasonal_2'].permute(0, 2, 1).contiguous()
        x23 = tmpq_dict2['seasonal_3'].permute(0, 2, 1).contiguous()
        # print(x2t.shape)            # torch.Size([896, 168, 1])
        # print(x21.shape)            # torch.Size([896, 168, 1])
        # print(x22.shape)            # torch.Size([896, 168, 1])
        # print(x23.shape)            # torch.Size([896, 168, 1])

        tmpq_dict3 = TMPQ(x3)
        x3t = tmpq_dict3['trend'].permute(0, 2, 1).contiguous()
        x31 = tmpq_dict3['seasonal_1'].permute(0, 2, 1).contiguous()
        x32 = tmpq_dict3['seasonal_2'].permute(0, 2, 1).contiguous()
        x33 = tmpq_dict3['seasonal_3'].permute(0, 2, 1).contiguous()
        # print(x3t.shape)            # torch.Size([896, 84, 1])
        # print(x31.shape)            # torch.Size([896, 84, 1])
        # print(x32.shape)            # torch.Size([896, 84, 1])
        # print(x33.shape)            # torch.Size([896, 84, 1])

        tmpq_dict4 = TMPQ(x4)
        x4t = tmpq_dict4['trend'].permute(0, 2, 1).contiguous()
        x41 = tmpq_dict4['seasonal_1'].permute(0, 2, 1).contiguous()
        x42 = tmpq_dict4['seasonal_2'].permute(0, 2, 1).contiguous()
        x43 = tmpq_dict4['seasonal_3'].permute(0, 2, 1).contiguous()
        # print(x4t.shape)            # torch.Size([896, 42, 1])
        # print(x41.shape)            # torch.Size([896, 42, 1])
        # print(x42.shape)            # torch.Size([896, 42, 1])
        # print(x43.shape)            # torch.Size([896, 42, 1])

        # print(level_embed.shape)    # torch.Size([3, 16])

        e1t = self.enc_embedding_1(x1t, None) + level_embed[0]
        e11 = self.enc_embedding_1(x11, None) + level_embed[1]
        e12 = self.enc_embedding_1(x12, None) + level_embed[2]
        e13 = self.enc_embedding_1(x13, None) + level_embed[3]

        e2t = self.enc_embedding_2(x2t, None) + level_embed[0]
        e21 = self.enc_embedding_2(x21, None) + level_embed[1]
        e22 = self.enc_embedding_2(x22, None) + level_embed[2]
        e23 = self.enc_embedding_2(x23, None) + level_embed[3]

        e3t = self.enc_embedding_3(x3t, None) + level_embed[0]
        e31 = self.enc_embedding_3(x31, None) + level_embed[1]
        e32 = self.enc_embedding_3(x32, None) + level_embed[2]
        e33 = self.enc_embedding_3(x33, None) + level_embed[3]

        e4t = self.enc_embedding_4(x4t, None) + level_embed[0]
        e41 = self.enc_embedding_4(x41, None) + level_embed[1]
        e42 = self.enc_embedding_4(x42, None) + level_embed[2]
        e43 = self.enc_embedding_4(x43, None) + level_embed[3]

        c1 = torch.concat([e1t, e11, e12, e13], dim=2)
        c2 = torch.concat([e2t, e21, e22, e23], dim=2)
        c3 = torch.concat([e3t, e31, e32, e33], dim=2)
        c4 = torch.concat([e4t, e41, e42, e43], dim=2)
        # print(c1.shape)             # torch.Size([896, 336, 64])
        # print(c2.shape)             # torch.Size([896, 168, 64])
        # print(c3.shape)             # torch.Size([896, 84, 64])
        # print(c4.shape)             # torch.Size([896, 42, 64])
        c_list = [c1, c2, c3, c4]

        # print('############# TMPTimeMixerEmbedding-2')
        return c_list


# 输入:
# c4(896,42,64),c3(896,84,64),c2(896,168,64),c1(896,336,64)
# 输出:
# dec_out(896,96)
class DecodeHeadTimeMixer(nn.Module):
    def __init__(self, token_num_max, d_model, pred_len, pool_scales=(1, 2, 3, 6), period_num=4,):
        super(DecodeHeadTimeMixer, self).__init__()
        self.d_model = d_model
        self.period_num = period_num
        self.pred_len = pred_len
        self.token_num_max = token_num_max
        self.d_model_out = 8

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
        self.conv_seg = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model_out, kernel_size=(1,))

        # Output
        self.flatten = nn.Flatten(start_dim=-2)
        self.output = nn.Linear(in_features=int(self.token_num_max * self.d_model_out), out_features=self.pred_len, bias=True)

        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        # print('############# decode_head-1')

        # 1. Conv-based Build Lateral.1-2
        # print(len(self.lateral_convs))          # 3
        # print(len(inputs))                      # 4
        laterals = []
        for i in range(len(self.lateral_convs)):
            lateral_conv = self.lateral_convs[i]
            input = inputs[i]
            lateral = lateral_conv(input.transpose(1, 2)).transpose(1, 2)
            # print(type(lateral_conv))           # <class 'torch.nn.modules.conv.Conv1d'>
            # print(input.shape)                  # torch.Size([896, 42, 64])     torch.Size([896, 84, 64])     torch.Size([896, 168, 64])
            # print(lateral.shape)                # torch.Size([896, 42, 64])     torch.Size([896, 84, 64])     torch.Size([896, 168, 64])
            laterals.append(lateral)

        # 2. PPM_based Build Lateral.3
        # print(inputs[-1].shape)                 # torch.Size([896, 336, 64])
        tmp = inputs[-1].transpose(1, 2)
        # print(tmp.shape)                        # torch.Size([896, 64, 336])
        psp_outs = [tmp]
        psp_out = self.psp_modules(tmp)
        # print(type(self.psp_modules))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.PPM'>
        # print(tmp.shape)                        # torch.Size([896, 64, 336])
        # print(len(psp_out))                     # 4
        # for tmp in psp_out:
        #     print(tmp.shape)                    # (896,64,336)    (896,64,336)    (896,64,336)    (896,64,336)
        psp_outs.extend(psp_out)
        psp_outs = torch.cat(psp_outs, dim=1)
        # print(psp_outs.shape)                   # torch.Size([896, 320, 336])

        psp_out = self.bottleneck(psp_outs)
        # print(type(self.bottleneck))            # <class 'torch.nn.modules.conv.Conv1d'>
        # print(psp_outs.shape)                   # torch.Size([896, 240, 336])
        # print(psp_out.shape)                    # torch.Size([896, 64, 336])
        psp_out = psp_out.transpose(1, 2)
        # print(psp_out.shape)                    # torch.Size([896, 336, 64])
        laterals.append(psp_out)

        # 3. FPN Top-Down Path, Lateral.1-3 Fusion
        # print(len(laterals))                    # 4
        # for tmp in laterals:
        #     print(tmp.shape)                    # (896,42,64)   (896,84,64)   (896,168,64)    (896,336,64)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1                              2
            # print(laterals[i].shape)            # torch.Size([896, 42, 64])     torch.Size([896, 84, 64])      torch.Size([896, 168, 64])
            lateral_tmp = F.interpolate(laterals[i].transpose(1, 2), scale_factor=laterals[i+1].shape[1]/laterals[i].shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
            # print(lateral_tmp.shape)            # torch.Size([896, 84, 64])     torch.Size([896, 168, 64])     torch.Size([896, 336, 64])
            # print(laterals[i + 1].shape)        # torch.Size([896, 84, 64])     torch.Size([896, 168, 64])     torch.Size([896, 336, 64])
            laterals[i + 1] = laterals[i + 1] + lateral_tmp
            # print(laterals[i + 1].shape)        # torch.Size([896, 84, 64])     torch.Size([896, 168, 64])     torch.Size([896, 336, 64])

        # 4. ShapeKeep Conv, Lateral.1-2
        # print(used_backbone_levels)             # 4
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1                             2
            layer = self.fpn_convs[i]
            lateral = laterals[i]
            fpn_out = layer(lateral.transpose(1, 2)).transpose(1, 2)
            # print(type(layer))                  # <class 'torch.nn.modules.conv.Conv1d'>
            # print(lateral.shape)                # torch.Size([896, 42, 64])     torch.Size([896, 84, 64])     torch.Size([896, 168, 64])
            # print(fpn_out.shape)                # torch.Size([896, 42, 64])     torch.Size([896, 84, 64])     torch.Size([896, 168, 64])
            fpn_outs.append(fpn_out)

        # print(laterals[-1].shape)               # torch.Size([896, 336, 64])
        fpn_outs.append(laterals[-1])

        # 5. Reshape: Lateral.2-3 -> Lateral.1
        # print(used_backbone_levels)             # 4
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1                             2
            # print(fpn_outs[i].shape)            # torch.Size([896, 42, 64])     torch.Size([896, 84, 64])     torch.Size([896, 168, 64])
            fpn_outs[i] = F.interpolate(fpn_outs[i].transpose(1, 2), scale_factor=fpn_outs[-1].shape[1]/fpn_outs[i].shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
            # print(fpn_outs[i].shape)            # torch.Size([896, 336, 64])    torch.Size([896, 336, 64])    torch.Size([896, 336, 64])

        # 6. Concat Lateral.1-3 DownConv Bottleneck
        # print(len(fpn_outs))                    # 4
        # for fpn_out in fpn_outs:
        #     print(fpn_out.shape)                # torch.Size([896, 336, 64])    torch.Size([896, 336, 64])    torch.Size([896, 336, 64])
        fpn_outs = torch.cat(fpn_outs, dim=2)
        # print(fpn_outs.shape)                   # torch.Size([896, 336, 256])

        feat = self.fpn_bottleneck(fpn_outs.transpose(1, 2)).transpose(1, 2)
        # print(type(self.fpn_bottleneck))        # <class 'torch.nn.modules.conv.Conv1d'>
        # print(fpn_outs.shape)                   # torch.Size([896, 336, 192])
        # print(feat.shape)                       # torch.Size([896, 336, 64])
        if self.dropout is not None:
            feat = self.dropout(feat)
        seasonal = self.conv_seg(feat.transpose(1, 2)).transpose(1, 2)
        # print(seasonal.shape)                   # torch.Size([896, 336, 8])
        seasonal = self.output(self.flatten(seasonal))
        # print(seasonal.shape)                   # torch.Size([896, 96])

        # print('############# decode_head-2')
        return seasonal
