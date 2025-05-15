import math
from argparse import Namespace
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_
from huggingface_hub import PyTorchModelHubMixin

from models.moment_module import Masking
from models.moment import Model
from adapter_modules._for_Moment import AdapterMomentBlock, TMPMomentEmbedding, DecodeHeadMoment


def model_show(model):
    show = True

    if show:
        print('#########################################################')

        print('')
        print('With Grad Parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        print('')
        print('Without Grad Parameters:')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(name)

        print('#########################################################')
    else:
        return None


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


class Model_Adapter(Model):
    def __init__(
            self,
            config, adapter_task_name, pred_len, n_channels, num_class,         # Backbone独有
            cffn_ratio=2.0, init_values=0., dim_ratio=2.0, period_num=3,    # Adapter独有
            **kwargs: dict
    ):
        super().__init__(config, adapter_task_name, pred_len, n_channels, num_class, **kwargs)
        self.task_name = adapter_task_name
        self.pred_len = pred_len
        self.enc_in = n_channels
        self.num_class = num_class
        # print('############# Moment_Adapter-1')
        # ######################## Original Module

        # print(self.task_name)                                       # long_term_forecast
        # print(self.seq_len)                                         # 512
        # print(self.patch_len)                                       # 8

        # 0. 无需训练
        # 0.1 可逆实例规范化Reversible Instance Normalization, ReIN
        # print(self.config.getattr("revin_affine", False))           # False
        # print(type(self.normalizer))                                # <class 'models.moment_module.RevIN'>
        # 0.2 PatchTokenizer
        # print(self.config.patch_len)                                # 8
        # print(self.config.patch_stride_len)                         # 8
        # print(type(self.tokenizer))                                 # <class 'models.moment_module.Patching'>
        # 0.3 MaskGenerator
        # print(self.config.getattr("mask_ratio", 0.0))               # 0.0
        # print(type(self.mask_generator))                            # <class 'models.moment_module.Masking'>

        # 1. 需要训练
        # 1.1 PatchEmbedding
        # print(self.config.d_model)                                  # 1024
        # print(self.config.patch_len)                                # 8
        # print(self.config.patch_stride_len)                         # 8
        # print(self.config.getattr("patch_dropout", 0.1))            # 0.1
        # print(self.config.getattr("add_positional_embedding", True))# True
        # print(self.config.getattr("value_embedding_bias", False))   # False
        # print(self.config.getattr("orth_gain", 1.41))               # 1.41
        # print(type(self.patch_embedding))                           # <class 'models.moment_module.PatchEmbedding'>
        # 1.2 EncoderOnly Backbone
        # print(type(self.encoder))                                   # <class 'transformers.models.t5.modeling_t5.T5Stack'>
        # 1.3 Output Head
        # print(type(self.head))                                      # <class 'models.moment.PretrainHead'>

        # ######################## Plugin Module

        # 0. 预训练Backbone的模型参数

        self.n_heads = 16
        self.drop = 0.1

        self.adapter_layer_num = period_num+1
        self.backbone_layer_num = len(self.encoder.block)
        assert self.backbone_layer_num > self.adapter_layer_num, f"The Layer Num of Backbone ({self.backbone_layer_num}) is less than Adapter ({self.adapter_layer_num})"
        split_index_list = split_integer(self.backbone_layer_num, self.adapter_layer_num)
        self.interaction_indexes = [[sum(split_index_list[:i]), sum(split_index_list[:i+1])-1] for i in range(self.adapter_layer_num)]
        # print(self.backbone_layer_num)      # 24
        # print(self.adapter_layer_num)       # 4
        # print(split_index_list)             # [6, 6, 6, 6]
        # print(self.interaction_indexes)     # [[0, 5], [6, 11], [12, 17], [18, 23]]

        self.patch_len_1 = int(self.config.patch_len * 4)
        self.patch_len_2 = int(self.config.patch_len * 2)
        self.patch_len_3 = int(self.config.patch_len)
        self.stride_1 = self.patch_len_1
        self.stride_2 = self.patch_len_2
        self.stride_3 = self.patch_len_3
        self.token_num_1 = int(self.seq_len / self.stride_1)
        self.token_num_2 = int(self.seq_len / self.stride_2)
        self.token_num_3 = int(self.seq_len / self.stride_3)
        # print(self.patch_len_1)             # 32
        # print(self.patch_len_2)             # 16
        # print(self.patch_len_3)             # 8
        # print(self.token_num_1)             # 16
        # print(self.token_num_2)             # 32
        # print(self.token_num_3)             # 64

        # 1. SPM, 用于生成multi-scale的c
        self.spm = TMPMomentEmbedding(
            d_model=self.config.d_model,
            patch_len_1=self.patch_len_1, patch_len_2=self.patch_len_2, patch_len_3=self.patch_len_3,
            stride_1=self.stride_1, stride_2=self.stride_2, stride_3=self.stride_3,
            patch_num_1=self.token_num_1, patch_num_2=self.token_num_2, patch_num_3=self.token_num_3,
        )
        self.spm.apply(self._init_weights)
        # print(type(self.spm))               # <class 'adapter_modules._for_Moment.TMPMomentEmbedding'>
        # print(self.config.d_model)          # 1024

        # 2. Multi-Level Embedding, 用于为multi-scale的c添加层级嵌入信息
        self.level_embed = nn.Parameter(torch.zeros(period_num, self.config.d_model))
        normal_(self.level_embed)
        # print(self.level_embed.shape)       # torch.Size([3, 1024])
        # print(self.config.d_model)          # 1024

        # 3. 基于BackboneBlock封装得到的AdapterBlock, 其中负责将c和x融合，并
        self.interactions = nn.Sequential(*[
            AdapterMomentBlock(
                dim=self.config.d_model, num_heads=self.n_heads, cffn_ratio=cffn_ratio, drop=self.drop,
                init_values=init_values, dim_ratio=dim_ratio, extra_CTI=(True if i == len(self.interaction_indexes) - 1 else False)
            )
            for i in range(len(self.interaction_indexes))
        ])
        self.interactions.apply(self._init_weights)
        # print(type(AdapterMomentBlock))     # <class 'type'>
        # print(self.config.d_model)          # 1024
        # print(self.n_heads)                 # 16
        # print(cffn_ratio)                   # 2.0
        # print(self.drop)                    # 0.1
        # print(init_values)                  # 0.0
        # print(dim_ratio)                    # 2.0
        # print(self.interaction_indexes)     # [[0, 5], [6, 11], [12, 17], [18, 23]]

        # 4. Decode Head
        self.norm1 = nn.LayerNorm(self.config.d_model)
        self.norm2 = nn.LayerNorm(self.config.d_model)
        self.norm3 = nn.LayerNorm(self.config.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.flatten = nn.Flatten(start_dim=-2)
            self.up_xt = nn.Linear(in_features=int(self.token_num_3 * self.config.d_model), out_features=self.pred_len, bias=True)
            self.up_xt.apply(self._init_weights)
            self.up_ct = nn.Linear(in_features=self.seq_len, out_features=self.pred_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.head_ppm = DecodeHeadMoment(token_num_max=self.token_num_3, d_model=self.config.d_model, pred_len=self.pred_len)

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.flatten = nn.Flatten(start_dim=-2)
            self.up_xt = nn.Linear(in_features=int(self.token_num_3 * self.config.d_model), out_features=self.seq_len, bias=True)
            self.up_xt.apply(self._init_weights)
            self.up_ct = nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.head_ppm = DecodeHeadMoment(token_num_max=self.token_num_3, d_model=self.config.d_model, pred_len=self.seq_len)

        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.normt = nn.LayerNorm(self.config.d_model)
            self.down_token_num = nn.Conv1d(
                in_channels=(self.token_num_1 + self.token_num_2 + self.token_num_3*2),
                out_channels=self.token_num_3,
                kernel_size=(3,), stride=(1,), padding=(1,), bias=True
            )
            self.down_dim = nn.Conv1d(
                in_channels=self.config.d_model,
                out_channels=int(self.config.d_model/8),
                kernel_size=(3,), stride=(1,), padding=(1,), bias=True
            )
            self.projection = nn.Linear(
                in_features=self.token_num_3 * int(self.config.d_model/8) * self.enc_in,
                out_features=self.num_class
            )

        # 展示模型中【可训练参数】和【冻结部分参数】
        # model_show(self)
        """
        ########################## With Grad Parameters:
        level_embed
        head
        spm
        interactions
        norm1
        norm2
        norm3
        up_xt
        up_ct
        head_ppm
        ########################## Without Grad Parameters:
        patch_embedding
        encoder
        """
        # print('############# Moment_Adapter-2')

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

    # outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
    # outputs = self.backbone_forward(inputs_embeds=enc_in, attention_mask=attention_mask)
    def backbone_forward(self, inputs_embeds, attention_mask):
        from torch.utils.checkpoint import checkpoint
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

        use_cache = self.encoder.config.use_cache
        output_attentions = self.encoder.config.output_attentions
        # print(use_cache)                            # False
        # print(output_attentions)                    # False

        input_shape = inputs_embeds.size()[:-1]
        # print(inputs_embeds.shape)                  # torch.Size([224, 64, 1024])
        # print(input_shape)                          # torch.Size([224, 64])
        hidden_states = self.encoder.dropout(inputs_embeds)
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])

        # print(attention_mask.shape)                 # torch.Size([224, 64])
        # print(input_shape)                          # torch.Size([224, 64])
        attention_mask = self.encoder.get_extended_attention_mask(attention_mask, input_shape)
        # print(attention_mask.shape)                 # torch.Size([224, 1, 1, 64])

        # print(self.encoder.gradient_checkpointing)  # True
        # print(self.encoder.training)                # True

        # 初始化
        # print(len(self.encoder.block))              # 24
        position_bias = None
        for i, layer_module in enumerate(self.encoder.block):
            if self.encoder.gradient_checkpointing and self.encoder.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))
                    return custom_forward
                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    position_bias,

                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                )
            # print(i)                                # 0
            # print(type(layer_module))               # <class 'transformers.models.t5.modeling_t5.T5Block'>
            # print(hidden_states.shape)              # torch.Size([224, 64, 1024])
            # print(extended_attention_mask.shape)    # torch.Size([224, 1, 1, 64])
            # print(position_bias if i == 0 else
            #       position_bias.shape)              # None

            # print(len(layer_outputs))               # 2
            hidden_states, position_bias = layer_outputs[:2]
            # print(hidden_states.shape)              # torch.Size([224, 64, 1024])
            # print(position_bias.shape)              # torch.Size([224, 16, 64, 64])

        # print(type(self.encoder.final_layer_norm))  # <class 'transformers.models.t5.modeling_t5.T5LayerNorm'>
        # print(type(self.encoder.dropout))           # <class 'torch.nn.modules.dropout.Dropout'>
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])
        hidden_states = self.encoder.final_layer_norm(hidden_states)
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])
        hidden_states = self.encoder.dropout(hidden_states)
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def forecast(self, x_enc, input_mask):
        # print('############# ModelAdapter.LongForecasting-1')
        batch_size, seq_len, n_channels = x_enc.shape

        # 0. 基于RevIN实现历史序列的【可逆实例规范化】
        # (batch_size, n_channels, seq_len)
        # print(type(self.normalizer))        # <class 'models.moment_module.RevIN'>
        # print(x_enc.shape)                  # torch.Size([32, 512, 7])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        # print(input_mask.shape)             # torch.Size([32, 512])
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])

        # 1. SPM forward
        # SPM即Spatial Pyramid Matching, 是一种利用空间金字塔进行图像匹配、识别、分类的算法, SPM在不同分辨率上统计图像特征点分布，从而获取图像的局部信息。
        # print(type(self.spm))               # <class 'adapter_modules._for_Moment.TMPMomentEmbedding'>
        mask = torch.ones_like(input_mask).to(input_mask.device)
        # print(mask.shape)                   # torch.Size([32, 512])
        ct, c1, c2, c3 = self.spm(x_enc, mask)
        # print(ct.shape)                     # torch.Size([224, 512])
        # print(c1.shape)                     # torch.Size([224, 16, 1024])
        # print(c2.shape)                     # torch.Size([224, 32, 1024])
        # print(c3.shape)                     # torch.Size([224, 64, 1024])

        # 2. Multi-Level Embedding
        # print(self.level_embed[0].shape)    # torch.Size([1024])
        # print(self.level_embed[1].shape)    # torch.Size([1024])
        # print(self.level_embed[2].shape)    # torch.Size([1024])
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        # print(c1.shape)                     # torch.Size([224, 16, 1024])
        # print(c2.shape)                     # torch.Size([224, 32, 1024])
        # print(c3.shape)                     # torch.Size([224, 64, 1024])
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)                      # torch.Size([224, 112, 1024])

        # 3. PatchEmbedding
        # 3.1 基于一种无需训练的方式, 直接将长度为512的历史序列拆分为64个patch, 每个patch长度为8
        # (batch_size, n_channels, seq_len) -> (batch_size, n_channels, patch_num, patch_len)
        # print(type(self.tokenizer))         # <class 'models.moment_module.Patching'>
        # print(self.config.patch_len)        # 8
        # print(self.config.patch_stride_len) # 8
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        x_enc = self.tokenizer(x=x_enc)
        # print(x_enc.shape)                  # torch.Size([32, 7, 64, 8])
        # 3.2 将patch_len(8)映射为d_model(1024),
        # (batch_size, n_channels, patch_num, patch_len) -> (batch_size, n_channels, patch_num, d_model)
        # print(type(self.patch_embedding))   # <class 'models.moment_module.PatchEmbedding'>
        # print(x_enc.shape)                  # torch.Size([32, 7, 64, 8])
        # print(input_mask.shape)             # torch.Size([32, 512])
        mask = torch.ones_like(input_mask).to(input_mask.device)
        # print(mask.shape)                   # torch.Size([32, 512])
        enc_in = self.patch_embedding(x_enc, mask=mask)
        # print(enc_in.shape)                 # torch.Size([32, 7, 64, 1024])
        # 3.3 ChannelIndependence
        # (batch_size, n_channels, patch_num, d_model) -> (batch_size*n_channels, patch_num, d_model)
        enc_out = enc_in.reshape((batch_size * n_channels, enc_in.shape[2], self.config.d_model)).contiguous()
        # print(enc_out.shape)                # torch.Size([224, 64, 1024])

        # 4. 为Encoder架构生成attention_mask, 其中每个样本的每个token对应一个值, 通道间共有mask值
        # (batch_size*n_channels, patch_num)
        # print(input_mask.shape)             # torch.Size([32, 512])
        # print(self.patch_len)               # 8
        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        # print(patch_view_mask.shape)        # torch.Size([32, 64])
        # print(n_channels)                   # 7
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        # print(attention_mask.shape)         # torch.Size([224, 64])

        # 5. Backbone & Adapter
        # print(len(self.interactions))       # 4
        # print(len(self.encoder.block))      # 24
        outs = []
        position_bias = None
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            # print(indexes)                  # [0, 5]            [6, 11]              [12, 17]                  [18, 23]
            adapter_block = self.interactions[i]
            backbone_blocks = self.encoder.block[indexes[0]:indexes[-1] + 1]
            # print(type(adapter_block))      # <class 'adapter_modules._for_Moment.AdapterMomentBlock'>
            # print(len(backbone_blocks))     # 6
            # print(type(backbone_blocks[0])) # <class 'transformers.models.t5.modeling_t5.T5Block'>
            # print(enc_out.shape)            # torch.Size([224, 64, 1024])
            # print(c.shape)                  # torch.Size([224, 112, 1024])
            enc_out, c, position_bias = adapter_block(
                x=enc_out, c=c, backbone_blocks=backbone_blocks, idxs=[c1.shape[1], c1.shape[1]+c2.shape[1]],
                encoder=self.encoder, attention_mask=attention_mask, position_bias=position_bias
            )
            # print(enc_out.shape)            # torch.Size([224, 64, 1024])
            # print(c.shape)                  # torch.Size([224, 112, 1024])
            # print(position_bias.shape)      # torch.Size([224, 16, 64, 64])
            outs.append(enc_out)

        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([224, 64, 1024])
        # print(x1.shape)                     # torch.Size([224, 64, 1024])
        # print(x2.shape)                     # torch.Size([224, 64, 1024])
        # print(x3.shape)                     # torch.Size([224, 64, 1024])

        # 5.1 Split Multi-period condition (MP-cond)
        # print(c.shape)                      # torch.Size([224, 112, 1024])
        c1 = c[:, 0:c1.shape[1], :]
        c2 = c[:, c1.shape[1]:c1.shape[1]+c2.shape[1], :]
        c3 = c[:, c1.shape[1]+c2.shape[1]:, :]
        # print(c1.shape)                     # torch.Size([224, 16, 1024])
        # print(c2.shape)                     # torch.Size([224, 32, 1024])
        # print(c3.shape)                     # torch.Size([224, 64, 1024])

        # 5.2 Fusion Multi-scale hidden feature from Adapter to Trend and Cond1,2,3
        x1 = F.interpolate(x1.transpose(1, 2).contiguous(), scale_factor=c1.shape[1]/x1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        x2 = F.interpolate(x3.transpose(1, 2).contiguous(), scale_factor=c2.shape[1]/x2.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        f1 = self.norm1(c1+x1)
        f2 = self.norm2(c2+x2)
        f3 = self.norm3(c3+x3)
        # print(f1.shape)                     # torch.Size([224, 16, 1024])
        # print(f2.shape)                     # torch.Size([224, 32, 1024])
        # print(f3.shape)                     # torch.Size([224, 64, 1024])

        # 6.1 Down-to-Up Path Decoder
        dec_out = self.head_ppm([f1, f2, f3])
        # print(type(self.head_ppm))          # <class 'adapter_modules._for_Moment.DecodeHeadMoment'>
        # print(dec_out.shape)                # torch.Size([224, 96])

        # 6.2 TrendResid AutoRegression
        # print(ct.shape)                     # torch.Size([224, 512])
        ct = self.up_ct(ct)
        # print(ct.shape)                     # torch.Size([224, 96])

        # 6.3 Fusion
        # print(xt.shape)                     # torch.Size([224, 64, 1024])
        xt = self.up_xt(self.flatten(xt))
        # print(xt.shape)                     # torch.Size([224, 96])
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([224, 96])

        # 7. Inverse-ChannelIndependence & RevIN & Permute
        # print(dec_out.shape)                # torch.Size([224, 96])
        dec_out = torch.reshape(dec_out, (-1, n_channels, dec_out.shape[-1])).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 7, 96])
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        # print(dec_out.shape)                # torch.Size([32, 7, 96])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 96, 7])
        # print('############# ModelAdapter.LongForecasting-2')
        return dec_out

    def reconstruction(self, x_enc, input_mask, mask, is_imputation=True):
        def avgImputation(x, mask):
            x_imp = x

            for i in range(x.shape[0]):
                for j in range(x.shape[2]):
                    seq = x[i, :, j]
                    seq_mask = mask[i, :]

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

        # print('############# ModelAdapter.Reconstruction-1')
        batch_size, seq_len, n_channels = x_enc.shape

        # AvgImputation
        if is_imputation:
            # print(x_enc.shape)              # torch.Size([32, 512, 7])
            # print(input_mask.shape)         # torch.Size([32, 512])
            x_enc = avgImputation(x_enc, mask)
            # print(x_enc.shape)              # torch.Size([32, 512, 7])
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            # print(x_enc.shape)              # torch.Size([32, 7, 512])
        else:
            # print(x_enc.shape)              # torch.Size([32, 512, 7])
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            # print(x_enc.shape)              # torch.Size([32, 7, 512])

        # ChannelIndependence
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        # print(input_mask.shape)             # torch.Size([32, 512])
        x_enc = x_enc.reshape((-1, 1, seq_len)).contiguous()
        input_mask = input_mask.repeat_interleave(n_channels, axis=0)
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        # print(input_mask.shape)             # torch.Size([224, 512])

        # 0. 基于RevIN实现历史序列的【可逆实例规范化】
        # print(type(self.normalizer))        # <class 'models.moment_module.RevIN'>
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        # print(mask.shape)                   # torch.Size([224, 512])
        # print(input_mask.shape)             # torch.Size([224, 512])
        # print((mask*input_mask).shape)      # torch.Size([224, 512])
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])

        # 1. SPM forward
        # 2. Multi-Level Embedding
        ct, c1, c2, c3 = self.spm(x_enc, mask)
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        c = torch.cat([c1, c2, c3], dim=1)
        # print(ct.shape)                     # torch.Size([224, 512])
        # print(c1.shape)                     # torch.Size([224, 16, 1024])
        # print(c2.shape)                     # torch.Size([224, 32, 1024])
        # print(c3.shape)                     # torch.Size([224, 64, 1024])
        # print(c.shape)                      # torch.Size([224, 112, 1024])

        # 3. PatchEmbedding
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        x_enc = self.tokenizer(x=x_enc)
        # print(x_enc.shape)                  # torch.Size([224, 1, 64, 8])
        # print(mask.shape)                   # torch.Size([224, 512])
        enc_in = self.patch_embedding(x_enc, mask=mask)
        # print(enc_in.shape)                 # torch.Size([224, 1, 64, 1024])
        enc_out = enc_in.reshape((-1, enc_in.shape[-2], enc_in.shape[-1])).contiguous()
        # print(enc_out.shape)                # torch.Size([224, 64, 1024])

        # 4. 为Encoder架构生成attention_mask, 其中每个样本的每个token对应一个值, 通道间共有mask值
        # print(input_mask.shape)             # torch.Size([224, 512])
        attention_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        # print(attention_mask.shape)         # torch.Size([224, 64])

        # 5. Backbone & Adapter
        outs = []
        position_bias = None
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.encoder.block[indexes[0]:indexes[-1] + 1]
            enc_out, c, position_bias = adapter_block(
                x=enc_out, c=c, backbone_blocks=backbone_blocks, idxs=[c1.shape[1], c1.shape[1]+c2.shape[1]],
                encoder=self.encoder, attention_mask=attention_mask, position_bias=position_bias
            )
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([224, 64, 1024])
        # print(x1.shape)                     # torch.Size([224, 64, 1024])
        # print(x2.shape)                     # torch.Size([224, 64, 1024])
        # print(x3.shape)                     # torch.Size([224, 64, 1024])

        # 5.1 Split Multi-period condition (MP-cond)
        # 5.2 Fusion Multi-scale hidden feature from Adapter to Trend and Cond1,2,3
        c1 = c[:, 0:c1.shape[1], :]
        c2 = c[:, c1.shape[1]:c1.shape[1]+c2.shape[1], :]
        c3 = c[:, c1.shape[1]+c2.shape[1]:, :]
        x1 = F.interpolate(x1.transpose(1, 2).contiguous(), scale_factor=c1.shape[1]/x1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        x2 = F.interpolate(x3.transpose(1, 2).contiguous(), scale_factor=c2.shape[1]/x2.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        f1 = self.norm1(c1+x1)
        f2 = self.norm2(c2+x2)
        f3 = self.norm3(c3+x3)
        # print(f1.shape)                     # torch.Size([224, 16, 1024])
        # print(f2.shape)                     # torch.Size([224, 32, 1024])
        # print(f3.shape)                     # torch.Size([224, 64, 1024])

        # 6.1 Down-to-Up Path Decoder
        # 6.2 TrendResid AutoRegression
        # 6.3 Fusion
        dec_out = self.head_ppm([f1, f2, f3])
        ct = self.up_ct(ct)
        xt = self.up_xt(self.flatten(xt))
        dec_out = dec_out + xt + ct
        # print(dec_out.shape)                # torch.Size([224, 512])

        # 7. Inverse-ChannelIndependence & RevIN & Permute
        dec_out = self.normalizer(x=dec_out.unsqueeze(1), mode="denorm").squeeze(1)
        dec_out = torch.reshape(dec_out, (-1, n_channels, dec_out.shape[-1])).contiguous()
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 512, 7])
        # print('############# ModelAdapter.Reconstruction-2')
        return dec_out

    def classify(self, x_enc, input_mask):
        # print('############# ModelAdapter.Classification-1')
        batch_size, seq_len, n_channels = x_enc.shape
        # print(x_enc.shape)                  # torch.Size([32, 512, 7])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])

        # 1. SPM forward
        # 2. Multi-Level Embedding
        ct, c1, c2, c3 = self.spm(x_enc, input_mask)
        c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        c = torch.cat([c1, c2, c3], dim=1)
        # print(ct.shape)                     # torch.Size([224, 512])
        # print(c1.shape)                     # torch.Size([224, 16, 1024])
        # print(c2.shape)                     # torch.Size([224, 32, 1024])
        # print(c3.shape)                     # torch.Size([224, 64, 1024])
        # print(c.shape)                      # torch.Size([224, 112, 1024])

        # 3. PatchEmbedding
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        x_enc = self.tokenizer(x=x_enc)
        # print(x_enc.shape)                  # torch.Size([32, 7, 64, 8])
        # print(input_mask.shape)             # torch.Size([32, 512])
        enc_in = self.patch_embedding(x_enc, mask=input_mask)
        # print(enc_in.shape)                 # torch.Size([32, 7, 64, 1024])
        enc_out = enc_in.reshape((batch_size * n_channels, enc_in.shape[2], self.config.d_model)).contiguous()
        # print(enc_out.shape)                # torch.Size([224, 64, 1024])

        # 4. 为Encoder架构生成attention_mask, 其中每个样本的每个token对应一个值, 通道间共有mask值
        # print(input_mask.shape)             # torch.Size([32, 512])
        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        # print(patch_view_mask.shape)        # torch.Size([32, 64])
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        # print(attention_mask.shape)         # torch.Size([224, 64])

        # 5. Backbone & Adapter
        outs = []
        position_bias = None
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.encoder.block[indexes[0]:indexes[-1] + 1]
            enc_out, c, position_bias = adapter_block(
                x=enc_out, c=c, backbone_blocks=backbone_blocks, idxs=[c1.shape[1], c1.shape[1] + c2.shape[1]],
                encoder=self.encoder, attention_mask=attention_mask, position_bias=position_bias
            )
            outs.append(enc_out)
        xt, x1, x2, x3 = outs
        # print(xt.shape)                     # torch.Size([224, 64, 1024])
        # print(x1.shape)                     # torch.Size([224, 64, 1024])
        # print(x2.shape)                     # torch.Size([224, 64, 1024])
        # print(x3.shape)                     # torch.Size([224, 64, 1024])

        # 5.1 Split Multi-period condition (MP-cond)
        # 5.2 Fusion Multi-scale hidden feature from Adapter to Trend and Cond1,2,3
        c1 = c[:, 0:c1.shape[1], :]
        c2 = c[:, c1.shape[1]:c1.shape[1] + c2.shape[1], :]
        c3 = c[:, c1.shape[1] + c2.shape[1]:, :]
        x1 = F.interpolate(x1.transpose(1, 2).contiguous(), scale_factor=c1.shape[1] / x1.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        x2 = F.interpolate(x3.transpose(1, 2).contiguous(), scale_factor=c2.shape[1] / x2.shape[1], mode='linear', align_corners=False, recompute_scale_factor=True).transpose(1, 2).contiguous()
        f1 = self.norm1(c1 + x1)
        f2 = self.norm2(c2 + x2)
        f3 = self.norm3(c3 + x3)
        # print(f1.shape)                     # torch.Size([224, 16, 1024])
        # print(f2.shape)                     # torch.Size([224, 32, 1024])
        # print(f3.shape)                     # torch.Size([224, 64, 1024])

        # 6. Concatenate
        # print(ct.shape)                     # torch.Size([224, 512])
        ct = torch.reshape(ct, (-1, n_channels, ct.shape[-1])).contiguous()
        # print(ct.shape)                     # torch.Size([32, 7, 512])
        ct = self.tokenizer(x=ct)
        # print(ct.shape)                     # torch.Size([32, 7, 64, 8])
        ct = self.patch_embedding(ct, mask=input_mask)
        # print(ct.shape)                     # torch.Size([32, 7, 64, 1024])
        ct = ct.reshape((batch_size * n_channels, ct.shape[2], self.config.d_model)).contiguous()
        # print(ct.shape)                     # torch.Size([224, 64, 1024])
        ft = self.normt(ct + xt)
        # print(ft.shape)                     # torch.Size([224, 64, 1024])
        output = torch.concat([ft, f1, f2, f3], dim=1)
        # print(output.shape)                 # torch.Size([224, 240, 1024])

        # 6.1 Conv1d-based TokenNum Down
        # print(output.shape)                 # torch.Size([224, 240, 1024])
        output = self.down_token_num(output)
        # print(output.shape)                 # torch.Size([224, 64, 1024])

        # 6.2 Conv1d-based Dim Down
        # print(output.shape)                 # torch.Size([224, 64, 1024])
        output = self.down_dim(output.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # print(output.shape)                 # torch.Size([224, 64, 128])
        output = torch.reshape(output, (-1, n_channels, output.shape[-2], output.shape[-1])).contiguous()
        # print(output.shape)                 # torch.Size([32, 7, 64, 128])

        # 6.3 Embedding & Unsupervised
        # 对于每个样本(32)，每个通道(7)，包含64个patch，每个patch向量的长度为128
        # 我们希望将7个通道的patch向量拼接在一起，得到长度为896的patch向量，因此返回embedding(32,64,896)
        embedding = output.clone()
        # print(embedding.shape)              # torch.Size([32, 7, 64, 128])
        embedding = embedding.permute(0, 2, 3, 1).contiguous()
        # print(embedding.shape)              # torch.Size([32, 64, 128, 7])
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], -1).contiguous()

        # 6.4 Flatten Classification
        # print(output.shape)                 # torch.Size([32, 7, 64, 128])
        output = output.reshape(output.shape[0], -1).contiguous()
        # print(output.shape)                 # torch.Size([32, 57344])
        logits = self.projection(output)
        # print(logits.shape)                 # torch.Size([32, 5])
        # print('############# ModelAdapter.Classification-2')
        return logits, embedding

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).float()
            # print(input_mask.shape)             # torch.Size([32, 512])
            output = self.forecast(x_enc=x_enc, input_mask=input_mask)
            # print(output.shape)                 # torch.Size([32, 192, 7])
            return output

        elif self.task_name == 'imputation':
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).long()
            # print(input_mask.shape)             # torch.Size([32, 512])
            # print(mask.shape)                   # torch.Size([32, 512, 7])
            mask = mask.permute(0, 2, 1).contiguous()
            # print(mask.shape)                   # torch.Size([32, 7, 512])
            mask = mask.reshape((-1, mask.shape[-1])).long().contiguous()
            # print(mask.shape)                   # torch.Size([224, 512])
            output = self.reconstruction(x_enc=x_enc, input_mask=input_mask, mask=mask, is_imputation=True)
            # print(output.shape)                 # torch.Size([32, 512, 7])
            return output

        elif self.task_name == 'anomaly_detection':
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).long()
            # print(input_mask.shape)             # torch.Size([32, 512])
            # print(mask.shape)                   # torch.Size([32, 512, 7])
            mask = mask.permute(0, 2, 1).contiguous()
            # print(mask.shape)                   # torch.Size([32, 7, 512])
            mask = mask.reshape((-1, mask.shape[-1])).long().contiguous()
            # print(mask.shape)                   # torch.Size([224, 512])
            output = self.reconstruction(x_enc=x_enc, input_mask=input_mask, mask=mask, is_imputation=False)
            # print(output.shape)                 # torch.Size([32, 512, 7])
            return output

        # 在分类任务中，输入的时间序列样本往往不到512个时间步，因此需要在原始序列的前面填充零，
        # 与之对应地，input_mask为0代表是填充值、为1代表是真实值，因此对于预测插补异常检测的input_mask为全1张量
        elif self.task_name == 'classification':
            # 填充x_enc
            # print(x_enc.shape)                  # torch.Size([32, 140, 7])
            orig_len = x_enc.shape[1]
            # print(orig_len)                     # 140
            x_enc = torch.cat([
                torch.zeros((x_enc.shape[0], 512 - x_enc.shape[1], x_enc.shape[2])).to(x_enc.device).float(),
                x_enc,
            ], dim=1).to(x_enc.device).float()
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            # 填充input_mask
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).long()
            input_mask[:, :(x_enc.shape[1] - orig_len)] = 0
            # print(input_mask.shape)             # torch.Size([32, 512])
            logits, embedding = self.classify(x_enc=x_enc, input_mask=input_mask)
            # print(logits.shape)                 # torch.Size([32, 5])
            # print(embedding.shape)              # torch.Size([32, 64, 896])
            return logits

        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


# 确保具有足够的输入参数：
# 1. 对于forecasting任务, 必须指定forecast_horizon
# 2. 对于classification任务, 必须指定n_channels和num_class
class ModelPipeline_Adapter(Model_Adapter, PyTorchModelHubMixin):
    def __init__(self, config, **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        task_name = kwargs.get("model_kwargs", {}).pop("task_name")
        pred_len = kwargs.get("model_kwargs", {}).pop("pred_len", None)
        n_channels = kwargs.get("model_kwargs", {}).pop("n_channels", None)
        num_class = kwargs.get("model_kwargs", {}).pop("num_class", None)
        super().__init__(config=config, adapter_task_name=task_name, pred_len=pred_len, n_channels=n_channels, num_class=num_class, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == 'long_term_forecast' or config.task_name == 'short_term_forecast':
            if not hasattr(config, "pred_len"):
                raise ValueError("pred_len must be specified for long-horizon forecasting.")

        if config.task_name == 'classification':
            if not hasattr(config, "n_channels"):
                raise ValueError("n_channels must be specified for classification.")
            if not hasattr(config, "num_class"):
                raise ValueError("num_class must be specified for classification.")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_enc = torch.zeros((2, 512, 7)).to(device).float()

    # Forecasting
    model = ModelPipeline_Adapter.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'long_term_forecast', 'pred_len': 192},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    print(x_enc.shape)                  # torch.Size([32, 512, 7])
    output_forecasting = model(x_enc=x_enc)
    print(output_forecasting.shape)     # torch.Size([32, 192, 7])

    # Imputation
    model = ModelPipeline_Adapter.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'imputation'},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    mask = torch.rand(x_enc.shape).to(x_enc.device)
    mask[mask <= 0.3] = 0
    mask[mask > 0.3] = 1
    print(x_enc.shape)                  # torch.Size([32, 512, 7])
    print(mask.shape)                   # torch.Size([32, 512, 7])
    output_imputation = model(x_enc=x_enc, mask=mask)
    print(output_imputation.shape)      # torch.Size([32, 512, 7])

    # Anomaly_Detection
    model = ModelPipeline_Adapter.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "anomaly_detection"},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    mask = torch.rand(x_enc.shape).to(x_enc.device).long()
    mask[mask <= 0.3] = 0
    mask[mask > 0.3] = 1
    print(x_enc.shape)                  # torch.Size([32, 512, 7])
    print(mask.shape)                   # torch.Size([32, 512, 7])
    output_anomaly = model(x_enc=x_enc, mask=mask)
    print(output_anomaly.shape)         # torch.Size([32, 512, 7])

    # Classification
    # 在分类任务中，输入的时间序列样本往往不到512个时间步，因此需要在原始序列的前面填充零，
    # 与之对应地，input_mask为0代表是填充值、为1代表是真实值，因此对于预测插补异常检测的input_mask为全1张量
    x_enc = torch.zeros((2, 140, 7)).to(device).float()
    model = ModelPipeline_Adapter.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'classification', 'n_channels': 7, 'num_class': 5},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    print(x_enc.shape)                  # torch.Size([32, 140, 7])
    logits, embedding = model(x_enc=x_enc)
    print(logits.shape)                 # torch.Size([32, 5])
    print(embedding.shape)              # torch.Size([32, 64, 896])
