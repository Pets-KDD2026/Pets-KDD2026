import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter_modules.comer_modules_moment import MRFP, Extractor_CTI, CTI_toC, CTI_toV, PPM
from models.moment_module import PatchEmbedding, Patching

from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ


# print(c1.shape)             # torch.Size([224, 16, 1024])
# print(c2.shape)             # torch.Size([224, 32, 1024])
# print(c3.shape)             # torch.Size([224, 64, 1024])
# print(c.shape)              # torch.Size([224, 112, 1024])
# print(x.shape)              # torch.Size([224, 64, 1024])
class AdapterMomentBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=16, cffn_ratio=2.0, drop=0.1, init_values=0., extra_CTI=False, dim_ratio=2.0):
        super().__init__()

        self.cti_tov = CTI_toV(dim=dim, num_heads=num_heads, init_values=init_values, drop=drop, cffn_ratio=cffn_ratio)
        self.cti_toc = CTI_toC(dim=dim, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
        if extra_CTI:
            self.extra_CTIs = Extractor_CTI(dim=dim, num_heads=num_heads, cffn_ratio=cffn_ratio, drop=drop)
        else:
            self.extra_CTIs = None

        self.mrfp = MRFP(in_features=dim, hidden_features=int(dim * dim_ratio))

    def forward(
            self, x, c, backbone_blocks, idxs,
            encoder, attention_mask, position_bias
    ):
        # print('############# AdapterTemporalBlock-1')
        # print(c.shape)                      # torch.Size([224, 112, 1024])
        c = self.mrfp(c, idxs)
        # print(c.shape)                      # torch.Size([224, 112, 1024])

        c1 = c[:, 0:idxs[0], :].contiguous()
        c2 = c[:, idxs[0]:idxs[1], :].contiguous()
        c3 = c[:, idxs[1]:, :].contiguous()
        # print(c1.shape)                     # torch.Size([224, 16, 1024])
        # print(c2.shape)                     # torch.Size([224, 32, 1024])
        # print(c3.shape)                     # torch.Size([224, 64, 1024])
        # print(x.shape)                      # torch.Size([224, 64, 1024])
        c3 = c3 + x

        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)                      # torch.Size([224, 112, 1024])

        # print(type(self.cti_tov))           # <class 'adapter_modules.comer_modules.CTI_toV'>
        # print(x.shape)                      # torch.Size([224, 64, 1024])
        # print(c.shape)                      # torch.Size([224, 112, 1024])
        x = self.cti_tov(x=x, c=c, idxs=idxs)
        # print(x.shape)                      # torch.Size([224, 64, 1024])

        # print('###############################################')
        # print('############# AdapterTemporalBlock.T5Backbone-1')
        # print('###############################################')
        from torch.utils.checkpoint import checkpoint

        use_cache = encoder.config.use_cache
        output_attentions = encoder.config.output_attentions
        # print(use_cache)                    # False
        # print(output_attentions)            # False

        input_shape = x.size()[:-1]
        # print(x.shape)                      # torch.Size([224, 64, 1024])
        # print(input_shape)                  # torch.Size([224, 64])
        x = encoder.dropout(x)
        # print(x.shape)                      # torch.Size([224, 64, 1024])

        # print(attention_mask.shape)         # torch.Size([224, 64])
        # print(input_shape)                  # torch.Size([224, 64])
        attention_mask = encoder.get_extended_attention_mask(attention_mask, input_shape)
        # print(attention_mask.shape)         # torch.Size([224, 1, 1, 64])
        #
        # print(len(backbone_blocks))         # 6
        for i, backbone_block in enumerate(backbone_blocks):
            if encoder.gradient_checkpointing and encoder.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))
                    return custom_forward
                layer_outputs = checkpoint(
                    create_custom_forward(backbone_block),
                    x,
                    attention_mask,
                    position_bias,

                )
            else:
                layer_outputs = backbone_block(
                    x,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                )
            # print(type(backbone_block))     # <class 'transformers.models.t5.modeling_t5.T5Block'>
            # print(x.shape)                  # torch.Size([224, 64, 1024])
            # print(attention_mask.shape)     # torch.Size([224, 1, 1, 64])

            # print(len(layer_outputs))       # 2
            x, position_bias = layer_outputs[:2]
            # print(x.shape)                  # torch.Size([224, 64, 1024])
            # print(position_bias.shape)      # torch.Size([224, 16, 64, 64])

        # print(type(encoder.final_layer_norm))# <class 'transformers.models.t5.modeling_t5.T5LayerNorm'>
        # print(type(encoder.dropout))        # <class 'torch.nn.modules.dropout.Dropout'>
        # print(x.shape)                      # torch.Size([224, 64, 1024])
        x = encoder.final_layer_norm(x)
        # print(x.shape)                      # torch.Size([224, 64, 1024])
        x = encoder.dropout(x)
        # print(x.shape)                      # torch.Size([224, 64, 1024])

        # print('###############################################')
        # print('############# AdapterTemporalBlock.T5Backbone-2')
        # print('###############################################')

        # print(type(self.cti_toc))           # <class 'adapter_modules.comer_modules.CTI_toC'>
        # print(x.shape)                      # torch.Size([224, 64, 1024])
        # print(c.shape)                      # torch.Size([224, 112, 1024])
        c = self.cti_toc(x=x, c=c, idxs=idxs)
        # print(c.shape)                      # torch.Size([224, 112, 1024])

        # print(type(self.extra_CTIs))        # None    None    None    <class 'Adapter4TS_1D.adapter_modules.comer_modules.Extractor_CTI'>
        if self.extra_CTIs is not None:
            # print(x.shape)                  # torch.Size([224, 64, 1024])
            # print(c.shape)                  # torch.Size([224, 112, 1024])
            c = self.extra_CTIs(x=x, c=c, idxs=idxs)
            # print(c.shape)                  # torch.Size([224, 112, 1024])

        # print(x.shape)                      # torch.Size([224, 64, 1024])
        # print(c.shape)                      # torch.Size([224, 112, 1024])
        # print('############# AdapterTemporalBlock-2')
        return x, c, position_bias


# Spatial Pyramid Matching (SPM)
# 输入: c(32,7,5,512)
# 输出: ct(224,512)    c1(224,16,1024)    c2(224,32,1024)    c3(224,64,1024)    cr(224,512)
class TMPMomentEmbedding(nn.Module):
    def __init__(
            self, d_model,
            patch_len_1, patch_len_2, patch_len_3, stride_1, stride_2, stride_3, patch_num_1, patch_num_2, patch_num_3,
            patch_dropout=0.1, add_positional_embedding=True, value_embedding_bias=False, orth_gain=1.41
    ):
        super().__init__()

        # Patch Tokenizer
        self.tokenizer_1 = Patching(patch_len=patch_len_1, stride=stride_1)
        self.tokenizer_2 = Patching(patch_len=patch_len_2, stride=stride_2)
        self.tokenizer_3 = Patching(patch_len=patch_len_3, stride=stride_3)

        # Patch Embedding
        self.patch_embedding_1 = PatchEmbedding(
            d_model=d_model, patch_len=patch_len_1, stride=stride_1,
            patch_dropout=patch_dropout, add_positional_embedding=add_positional_embedding,
            value_embedding_bias=value_embedding_bias, orth_gain=orth_gain,
        )
        self.patch_embedding_2 = PatchEmbedding(
            d_model=d_model, patch_len=patch_len_2, stride=stride_2,
            patch_dropout=patch_dropout, add_positional_embedding=add_positional_embedding,
            value_embedding_bias=value_embedding_bias, orth_gain=orth_gain,
        )
        self.patch_embedding_3 = PatchEmbedding(
            d_model=d_model, patch_len=patch_len_3, stride=stride_3,
            patch_dropout=patch_dropout, add_positional_embedding=add_positional_embedding,
            value_embedding_bias=value_embedding_bias, orth_gain=orth_gain,
        )

        # Channel Embedding
        self.channel_embedding_1 = nn.Conv1d(patch_num_1, patch_num_1, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_2 = nn.Conv1d(patch_num_2, patch_num_2, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_embedding_3 = nn.Conv1d(patch_num_3, patch_num_3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # print('############# TMPMomentEmbedding-1')
        # print(x_enc.shape)          # torch.Size([32, 7, 512])          torch.Size([224, 1, 512])

        tmpq_dict = TMPQ(x)
        ct = tmpq_dict['trend'].squeeze(1)
        c1 = tmpq_dict['seasonal_1'].squeeze(1)
        c2 = tmpq_dict['seasonal_2'].squeeze(1)
        c3 = tmpq_dict['seasonal_3'].squeeze(1)
        # print(ct.shape)             # torch.Size([32, 7, 512])          torch.Size([224, 1, 512])
        # print(c1.shape)             # torch.Size([32, 7, 512])          torch.Size([224, 1, 512])
        # print(c2.shape)             # torch.Size([32, 7, 512])          torch.Size([224, 1, 512])
        # print(c3.shape)             # torch.Size([32, 7, 512])          torch.Size([224, 1, 512])

        c1 = self.tokenizer_1(c1)
        c2 = self.tokenizer_2(c2)
        c3 = self.tokenizer_3(c3)
        # print(c1.shape)             # torch.Size([32, 7, 16, 32])       torch.Size([224, 1, 16, 32])
        # print(c2.shape)             # torch.Size([32, 7, 32, 16])       torch.Size([224, 1, 32, 16])
        # print(c3.shape)             # torch.Size([32, 7, 64, 8])        torch.Size([224, 1, 64, 8])

        # print(mask.shape)           # torch.Size([32, 512])             torch.Size([224, 512])
        c1 = self.patch_embedding_1(c1, mask=mask)
        c2 = self.patch_embedding_2(c2, mask=mask)
        c3 = self.patch_embedding_3(c3, mask=mask)
        # print(c1.shape)             # torch.Size([32, 7, 16, 1024])     torch.Size([224, 1, 16, 1024])
        # print(c2.shape)             # torch.Size([32, 7, 32, 1024])     torch.Size([224, 1, 32, 1024])
        # print(c3.shape)             # torch.Size([32, 7, 64, 1024])     torch.Size([224, 1, 64, 1024])

        ct = torch.reshape(ct, (ct.shape[0] * ct.shape[1], ct.shape[2])).contiguous()
        c1 = torch.reshape(c1, (c1.shape[0] * c1.shape[1], c1.shape[2], c1.shape[3])).contiguous()
        c2 = torch.reshape(c2, (c2.shape[0] * c2.shape[1], c2.shape[2], c2.shape[3])).contiguous()
        c3 = torch.reshape(c3, (c3.shape[0] * c3.shape[1], c3.shape[2], c3.shape[3])).contiguous()
        # print(ct.shape)             # torch.Size([224, 512])            same
        # print(c1.shape)             # torch.Size([224, 16, 1024])       same
        # print(c2.shape)             # torch.Size([224, 32, 1024])       same
        # print(c3.shape)             # torch.Size([224, 64, 1024])       same
        # print(cr.shape)             # torch.Size([224, 512])            same

        c1 = self.channel_embedding_1(c1)
        c2 = self.channel_embedding_2(c2)
        c3 = self.channel_embedding_3(c3)
        # print(c1.shape)             # torch.Size([224, 16, 1024])       same
        # print(c2.shape)             # torch.Size([224, 32, 1024])       same
        # print(c3.shape)             # torch.Size([224, 64, 1024])       same

        # print('############# TMPMomentEmbedding-2')
        return ct, c1, c2, c3


# 输入: f1(224,16,1024), f2(224,32,1024), f3(224,64,1024)
# 输出: dec_out(224,96)
class DecodeHeadMoment(nn.Module):
    def __init__(self, token_num_max, d_model, pred_len, pool_scales=(1, 2, 3, 6), period_num=3,):
        super(DecodeHeadMoment, self).__init__()
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
            # print(input.shape)                  # torch.Size([224, 16, 1024])     torch.Size([224, 32, 1024])
            # print(lateral.shape)                # torch.Size([224, 16, 1024])     torch.Size([224, 32, 1024])
            laterals.append(lateral)

        # 2. PPM_based Build Lateral.3
        # print(inputs[-1].shape)                 # torch.Size([224, 64, 1024])
        tmp = inputs[-1].transpose(1, 2)
        # print(tmp.shape)                        # torch.Size([224, 1024, 64])
        psp_outs = [tmp]
        psp_out = self.psp_modules(tmp)
        # print(type(self.psp_modules))           # <class 'Adapter4TS_1D.adapter_modules.comer_modules.PPM'>
        # print(tmp.shape)                        # torch.Size([224, 1024, 64])
        # print(len(psp_out))                     # 4
        # for tmp in psp_out:
        #     print(tmp.shape)                    # (224,1024,64)    (224,1024,64)    (224,1024,64)    (224,1024,64)
        psp_outs.extend(psp_out)
        psp_outs = torch.cat(psp_outs, dim=1)
        # print(psp_outs.shape)                   # torch.Size([224, 5120, 64])

        psp_out = self.bottleneck(psp_outs)
        # print(type(self.bottleneck))            # <class 'torch.nn.modules.conv.Conv1d'>
        # print(psp_outs.shape)                   # torch.Size([224, 5120, 64])
        # print(psp_out.shape)                    # torch.Size([224, 1024, 64])
        psp_out = psp_out.transpose(1, 2)
        # print(psp_out.shape)                    # torch.Size([224, 64, 1024])
        laterals.append(psp_out)

        # 3. FPN Top-Down Path, Lateral.1-3 Fusion
        # print(len(laterals))                    # 3
        # for tmp in laterals:
        #     print(tmp.shape)                    # (224,16,1024)   (224,32,1024)   (224,64,1024)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            # print(laterals[i].shape)            # torch.Size([224, 16, 1024])   torch.Size([224, 32, 1024])
            lateral_tmp = F.interpolate(laterals[i].transpose(1, 2), scale_factor=laterals[i+1].shape[1]/laterals[i].shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
            # print(lateral_tmp.shape)            # torch.Size([224, 32, 1024])   torch.Size([224, 64, 1024])
            # print(laterals[i + 1].shape)        # torch.Size([224, 32, 1024])   torch.Size([224, 64, 1024])
            laterals[i + 1] = laterals[i + 1] + lateral_tmp
            # print(laterals[i + 1].shape)        # torch.Size([224, 32, 1024])   torch.Size([224, 64, 1024])

        # 4. ShapeKeep Conv, Lateral.1-2
        # print(used_backbone_levels)             # 3
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            layer = self.fpn_convs[i]
            lateral = laterals[i]
            fpn_out = layer(lateral.transpose(1, 2)).transpose(1, 2)
            # print(type(layer))                  # <class 'torch.nn.modules.conv.Conv1d'>
            # print(lateral.shape)                # torch.Size([224, 16, 1024])   torch.Size([224, 32, 1024])
            # print(fpn_out.shape)                # torch.Size([224, 16, 1024])   torch.Size([224, 32, 1024])
            fpn_outs.append(fpn_out)

        # print(laterals[-1].shape)               # torch.Size([224, 64, 1024])
        fpn_outs.append(laterals[-1])

        # 5. Reshape: Lateral.2-3 -> Lateral.1
        # print(used_backbone_levels)             # 3
        for i in range(used_backbone_levels - 1):
            # print(i)                            # 0                             1
            # print(fpn_outs[i].shape)            # torch.Size([224, 16, 1024])   torch.Size([224, 32, 1024])
            fpn_outs[i] = F.interpolate(fpn_outs[i].transpose(1, 2), scale_factor=fpn_outs[-1].shape[1]/fpn_outs[i].shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2)
            # print(fpn_outs[i].shape)            # torch.Size([224, 64, 1024])   torch.Size([224, 64, 1024])

        # 6. Concat Lateral.1-3 DownConv Bottleneck
        # print(len(fpn_outs))                    # 3
        # for fpn_out in fpn_outs:
        #     print(fpn_out.shape)                # torch.Size([224, 64, 1024])   torch.Size([224, 64, 1024])    torch.Size([224, 64, 1024])
        fpn_outs = torch.cat(fpn_outs, dim=2)
        # print(fpn_outs.shape)                   # torch.Size([224, 64, 3072])

        feat = self.fpn_bottleneck(fpn_outs.transpose(1, 2)).transpose(1, 2)
        # print(type(self.fpn_bottleneck))        # <class 'torch.nn.modules.conv.Conv1d'>
        # print(fpn_outs.shape)                   # torch.Size([224, 64, 3072])
        # print(feat.shape)                       # torch.Size([224, 64, 1024])
        if self.dropout is not None:
            feat = self.dropout(feat)
        seasonal = self.conv_seg(feat.transpose(1, 2)).transpose(1, 2)
        # print(seasonal.shape)                   # torch.Size([224, 64, 1024])
        seasonal = self.output(self.flatten(seasonal))
        # print(seasonal.shape)                   # torch.Size([224, 96])

        # print('############# decode_head-2')
        return seasonal

