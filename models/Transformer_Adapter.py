import os
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from adapter_modules.comer_modules import Normalize


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        from math import sqrt
        B, L, E = queries.shape
        _, S, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("ble,bse->bls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bls,bsd->bld", A, values)
        return V.contiguous(), A


class ModelAdapter(nn.Module):
    def __init__(self, configs, is_show=False):
        super(ModelAdapter, self).__init__()
        self.task_name = configs.task_name
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)

        # Pattern:1
        # Weights of AttnLayer
        self.query_projection_p1 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.key_projection_p1 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.value_projection_p1 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.out_projection_p1 = nn.Linear(configs.d_model * configs.enc_in, configs.enc_in)
        self.attention = FullAttention()
        # Weights of FFNLayer
        self.conv1_p1 = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.d_model * configs.enc_in, kernel_size=1)
        self.conv2_p1 = nn.Conv1d(in_channels=configs.d_model * configs.enc_in, out_channels=configs.enc_in, kernel_size=1)
        self.norm1_p1 = nn.LayerNorm(configs.enc_in)
        self.norm2_p1 = nn.LayerNorm(configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:2
        # Weights of AttnLayer
        self.query_projection_p2 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.key_projection_p2 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.value_projection_p2 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.out_projection_p2 = nn.Linear(configs.d_model * configs.enc_in, configs.enc_in)
        self.attention = FullAttention()
        # Weights of FFNLayer
        self.conv1_p2 = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.d_model * configs.enc_in, kernel_size=1)
        self.conv2_p2 = nn.Conv1d(in_channels=configs.d_model * configs.enc_in, out_channels=configs.enc_in, kernel_size=1)
        self.norm1_p2 = nn.LayerNorm(configs.enc_in)
        self.norm2_p2 = nn.LayerNorm(configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:3
        # Weights of AttnLayer
        self.query_projection_p3 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.key_projection_p3 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.value_projection_p3 = nn.Linear(configs.enc_in, configs.d_model * configs.enc_in)
        self.out_projection_p3 = nn.Linear(configs.d_model * configs.enc_in, configs.enc_in)
        self.attention = FullAttention()
        # Weights of FFNLayer
        self.conv1_p3 = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.d_model * configs.enc_in, kernel_size=1)
        self.conv2_p3 = nn.Conv1d(in_channels=configs.d_model * configs.enc_in, out_channels=configs.enc_in, kernel_size=1)
        self.norm1_p3 = nn.LayerNorm(configs.enc_in)
        self.norm2_p3 = nn.LayerNorm(configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:Trend and Projection:1,2,3
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection_p1 = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.projection_p1.apply(self._init_weights)
            self.projection_p2 = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.projection_p2.apply(self._init_weights)
            self.projection_p3 = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.projection_p3.apply(self._init_weights)
            self.up_ct = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'classification':
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.classifier = nn.Linear(4 * configs.enc_in * configs.seq_len, configs.num_class)
            self.classifier.apply(self._init_weights)

        # 实验分为两部分, 第一部分旨在展示"未解耦情况下观测序列中任意两个timestep之间的attn",
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 旨在计算保存每个batch中输入的x(32,336,7)和attn(32,336,336)
        # 第二部分旨在展示"解耦之后每个波动序列中任意两个timestep之间的attn", 此时直接加载步骤1中的
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 直接加载步骤1中的x(32,336,7), 将x解耦为多个波动序列, 计算并保存每个波动序列对应的attn(32,336,336)
        self.is_show = is_show
        self.save_path = f'./show_{configs.model}_v2/{configs.task_name}_{configs.model_id}'

    def flatten(self, x):
        from functools import partial
        original_shape = x.shape
        return x.flatten(), partial(np.reshape, newshape=original_shape)

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

    def encode_p1(self, x):
        # print('############# ShowModel.encode-1')
        # AttentionLayer
        # print(x.shape)                      # torch.Size([32, 336, 7])
        query = self.query_projection_p1(x)
        key = self.key_projection_p1(x)
        value = self.value_projection_p1(x)
        # print(query.shape)                  # torch.Size([32, 336, 21])
        # print(key.shape)                    # torch.Size([32, 336, 21])
        # print(value.shape)                  # torch.Size([32, 336, 21])
        new_x, attn = self.attention(
            query,
            key,
            value,
        )
        # print(new_x.shape)                  # torch.Size([32, 336, 21])
        # print(attn.shape)                   # torch.Size([32, 336, 336])
        new_x = self.out_projection_p1(new_x)
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = x + self.dropout(new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])

        # FFNLayer
        x = self.norm1_p1(x)
        new_x = x
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        new_x = self.dropout(self.activation(self.conv1_p1(new_x.transpose(-1, 1))))
        # print(new_x.shape)                  # torch.Size([32, 21, 336])
        new_x = self.dropout(self.conv2_p1(new_x).transpose(-1, 1))
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = self.norm2_p1(x + new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print('############# ShowModel.encode-2')
        return x, attn

    def encode_p2(self, x):
        # print('############# ShowModel.encode-1')
        # AttentionLayer
        # print(x.shape)                      # torch.Size([32, 336, 7])
        query = self.query_projection_p2(x)
        key = self.key_projection_p2(x)
        value = self.value_projection_p2(x)
        # print(query.shape)                  # torch.Size([32, 336, 21])
        # print(key.shape)                    # torch.Size([32, 336, 21])
        # print(value.shape)                  # torch.Size([32, 336, 21])
        new_x, attn = self.attention(
            query,
            key,
            value,
        )
        # print(new_x.shape)                  # torch.Size([32, 336, 21])
        # print(attn.shape)                   # torch.Size([32, 336, 336])
        new_x = self.out_projection_p2(new_x)
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = x + self.dropout(new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])

        # FFNLayer
        x = self.norm1_p2(x)
        new_x = x
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        new_x = self.dropout(self.activation(self.conv1_p2(new_x.transpose(-1, 1))))
        # print(new_x.shape)                  # torch.Size([32, 21, 336])
        new_x = self.dropout(self.conv2_p2(new_x).transpose(-1, 1))
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = self.norm2_p2(x + new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print('############# ShowModel.encode-2')
        return x, attn

    def encode_p3(self, x):
        # print('############# ShowModel.encode-1')
        # AttentionLayer
        # print(x.shape)                      # torch.Size([32, 336, 7])
        query = self.query_projection_p3(x)
        key = self.key_projection_p3(x)
        value = self.value_projection_p3(x)
        # print(query.shape)                  # torch.Size([32, 336, 21])
        # print(key.shape)                    # torch.Size([32, 336, 21])
        # print(value.shape)                  # torch.Size([32, 336, 21])
        new_x, attn = self.attention(
            query,
            key,
            value,
        )
        # print(new_x.shape)                  # torch.Size([32, 336, 21])
        # print(attn.shape)                   # torch.Size([32, 336, 336])
        new_x = self.out_projection_p3(new_x)
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = x + self.dropout(new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])

        # FFNLayer
        x = self.norm1_p3(x)
        new_x = x
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        new_x = self.dropout(self.activation(self.conv1_p3(new_x.transpose(-1, 1))))
        # print(new_x.shape)                  # torch.Size([32, 21, 336])
        new_x = self.dropout(self.conv2_p3(new_x).transpose(-1, 1))
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = self.norm2_p3(x + new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print('############# ShowModel.encode-2')
        return x, attn

    def forecast(self, x_enc, idx):
        # print('############# ShowModel.forecast-1')
        x = copy.deepcopy(x_enc)
        x = self.normalize_layer(x, 'norm')
        # print(x.shape)              # torch.Size([32, 336, 7])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)              # torch.Size([32, 7, 336])

        from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ
        tmpq_dict = TMPQ(x)
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        # print(ct.shape)             # torch.Size([32, 336, 7])
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(c3.shape)             # torch.Size([32, 336, 7])

        # Pattern:1
        # print(c1.shape)             # torch.Size([32, 336, 7])
        c1, attn1 = self.encode_p1(c1)
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(attn1.shape)          # torch.Size([32, 336, 336])
        o1 = self.projection_p1(c1.permute(0, 2, 1)).permute(0, 2, 1)
        # print(o1.shape)             # torch.Size([32, 96, 7])

        # Pattern:2
        # print(c2.shape)             # torch.Size([32, 336, 7])
        c2, attn2 = self.encode_p2(c2)
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(attn2.shape)          # torch.Size([32, 336, 336])
        o2 = self.projection_p2(c2.permute(0, 2, 1)).permute(0, 2, 1)
        # print(o2.shape)             # torch.Size([32, 96, 7])

        # Pattern:3
        # print(c3.shape)             # torch.Size([32, 336, 7])
        c3, attn3 = self.encode_p3(c3)
        # print(c3.shape)             # torch.Size([32, 336, 7])
        # print(attn3.shape)          # torch.Size([32, 336, 336])
        o3 = self.projection_p3(c3.permute(0, 2, 1)).permute(0, 2, 1)
        # print(o3.shape)             # torch.Size([32, 96, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 336, 7])
        ot = self.up_ct(ct.permute(0, 2, 1)).permute(0, 2, 1)
        # print(ot.shape)             # torch.Size([32, 96, 7])

        o = ot + o1 + o2 + o3
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 96, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern1.npy', attn1.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern2.npy', attn2.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern3.npy', attn3.cpu().detach().numpy())
        # print('############# ShowModel.forecast-2')
        return o

    def imputation(self, x_enc, idx):
        # print('############# ShowModel.imputation-1')
        x = copy.deepcopy(x_enc)
        x = self.normalize_layer(x, 'norm')
        # print(x.shape)              # torch.Size([32, 336, 7])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)              # torch.Size([32, 7, 336])

        from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ
        tmpq_dict = TMPQ(x)
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        # print(ct.shape)             # torch.Size([32, 336, 7])
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(c3.shape)             # torch.Size([32, 336, 7])

        # Pattern:1
        # print(c1.shape)             # torch.Size([32, 336, 7])
        o1, attn1 = self.encode_p1(c1)
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(attn1.shape)          # torch.Size([32, 336, 336])

        # Pattern:2
        # print(c2.shape)             # torch.Size([32, 336, 7])
        o2, attn2 = self.encode_p2(c2)
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(attn2.shape)          # torch.Size([32, 336, 336])

        # Pattern:3
        # print(c3.shape)             # torch.Size([32, 336, 7])
        o3, attn3 = self.encode_p3(c3)
        # print(c3.shape)             # torch.Size([32, 336, 7])
        # print(attn3.shape)          # torch.Size([32, 336, 336])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 336, 7])
        ot = self.up_ct(ct.permute(0, 2, 1)).permute(0, 2, 1)
        # print(ot.shape)             # torch.Size([32, 96, 7])

        o = ot + o1 + o2 + o3
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 96, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern1.npy', attn1.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern2.npy', attn2.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern3.npy', attn3.cpu().detach().numpy())
        # print('############# ShowModel.imputation-2')
        return o

    def anomaly_detection(self, x_enc, idx):
        # print('############# ShowModel.anomaly-1')
        x = copy.deepcopy(x_enc)
        x = self.normalize_layer(x, 'norm')
        # print(x.shape)              # torch.Size([32, 336, 7])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)              # torch.Size([32, 7, 336])

        from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ
        tmpq_dict = TMPQ(x)
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        # print(ct.shape)             # torch.Size([32, 336, 7])
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(c3.shape)             # torch.Size([32, 336, 7])

        # Pattern:1
        # print(c1.shape)             # torch.Size([32, 336, 7])
        o1, attn1 = self.encode_p1(c1)
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(attn1.shape)          # torch.Size([32, 336, 336])

        # Pattern:2
        # print(c2.shape)             # torch.Size([32, 336, 7])
        o2, attn2 = self.encode_p2(c2)
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(attn2.shape)          # torch.Size([32, 336, 336])

        # Pattern:3
        # print(c3.shape)             # torch.Size([32, 336, 7])
        o3, attn3 = self.encode_p3(c3)
        # print(c3.shape)             # torch.Size([32, 336, 7])
        # print(attn3.shape)          # torch.Size([32, 336, 336])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 336, 7])
        ot = self.up_ct(ct.permute(0, 2, 1)).permute(0, 2, 1)
        # print(ot.shape)             # torch.Size([32, 96, 7])

        o = ot + o1 + o2 + o3
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 96, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern1.npy', attn1.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern2.npy', attn2.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern3.npy', attn3.cpu().detach().numpy())
        # print('############# ShowModel.anomaly-2')
        return o

    def classification(self, x_enc, idx):
        # print('############# ShowModel.classification-1')
        x = copy.deepcopy(x_enc)
        # print(x.shape)              # torch.Size([32, 336, 7])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)              # torch.Size([32, 7, 336])

        from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ
        tmpq_dict = TMPQ(x)
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]].permute(0, 2, 1).contiguous()
        # print(ct.shape)             # torch.Size([32, 336, 7])
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(c3.shape)             # torch.Size([32, 336, 7])

        # Pattern:1
        # print(c1.shape)             # torch.Size([32, 336, 7])
        o1, attn1 = self.encode_p1(c1)
        # print(c1.shape)             # torch.Size([32, 336, 7])
        # print(attn1.shape)          # torch.Size([32, 336, 336])

        # Pattern:2
        # print(c2.shape)             # torch.Size([32, 336, 7])
        o2, attn2 = self.encode_p2(c2)
        # print(c2.shape)             # torch.Size([32, 336, 7])
        # print(attn2.shape)          # torch.Size([32, 336, 336])

        # Pattern:3
        # print(c3.shape)             # torch.Size([32, 336, 7])
        o3, attn3 = self.encode_p3(c3)
        # print(c3.shape)             # torch.Size([32, 336, 7])
        # print(attn3.shape)          # torch.Size([32, 336, 336])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 336, 7])
        ot = self.up_ct(ct.permute(0, 2, 1)).permute(0, 2, 1)
        # print(ot.shape)             # torch.Size([32, 336, 7])

        o = torch.cat([ot, o1, o2, o3], 2)
        # print(o.shape)              # torch.Size([32, 336, 28])

        # Output
        o = self.act(o)
        o = self.dropout(o)
        # print(o.shape)              # torch.Size([32, 336, 28])
        o = o.reshape(o.shape[0], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 9408])
        o = self.classifier(o)
        # print(o.shape)              # torch.Size([32, 4])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern1.npy', attn1.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern2.npy', attn2.cpu().detach().numpy())
            np.save(f'{self.save_path}/npy/{idx}_attn_pattern3.npy', attn3.cpu().detach().numpy())
        return o

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, idx=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, idx)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, idx)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, idx)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, idx)
            return dec_out


if __name__ == '__main__':
    """
    bash ./scripts/long_term_forecast/ETT_script/ShowModel/ETTh1_16.sh
    bash ./scripts/long_term_forecast/ETT_script/ShowModel/ETTh2_16.sh
    bash ./scripts/long_term_forecast/ETT_script/ShowModel/ETTm1_16.sh
    bash ./scripts/long_term_forecast/ETT_script/ShowModel/ETTm2_16.sh
    bash ./scripts/long_term_forecast/ECL_script/ShowModel.sh
    bash ./scripts/long_term_forecast/Solar_script/ShowModel.sh
    bash ./scripts/long_term_forecast/Traffic_script/ShowModel.sh
    bash ./scripts/long_term_forecast/Weather_script/ShowModel.sh

    bash ./scripts/imputation/ETT_script/ShowModel_ETTh1.sh
    bash ./scripts/imputation/ETT_script/ShowModel_ETTh2.sh
    bash ./scripts/imputation/ETT_script/ShowModel_ETTm1.sh
    bash ./scripts/imputation/ETT_script/ShowModel_ETTm2.sh
    bash ./scripts/imputation/ECL_script/ShowModel.sh
    bash ./scripts/imputation/Weather_script/ShowModel.sh

    bash ./scripts/anomaly_detection/ShowModel.sh
    bash ./scripts/classification/ShowModel.sh

    #########################################
    bash ./scripts/run_ShowModel.sh
    bash ./scripts/run_FreTS.sh
    #########################################



    """
    print('')
