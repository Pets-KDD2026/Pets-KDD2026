import os
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from adapter_modules.comer_modules import Normalize


class Model_v1(nn.Module):
    def __init__(self, configs):
        super(Model_v1, self).__init__()
        self.task_name = configs.task_name
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)
        self.sparsity_threshold = 0.01

        # 频域上的复数线性变换层
        self.scale = 0.02
        self.seq_len = configs.seq_len
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_len = configs.pred_len
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection' or 'classification':
            self.output_len = configs.seq_len
        self.freq_size_in = self.seq_len // 2 + 1
        self.freq_size_out = self.output_len // 2 + 1
        self.r = nn.Parameter(self.scale * torch.randn(self.freq_size_in, self.freq_size_out))
        self.i = nn.Parameter(self.scale * torch.randn(self.freq_size_in, self.freq_size_out))
        self.rb = nn.Parameter(self.scale * torch.randn(self.freq_size_out))
        self.ib = nn.Parameter(self.scale * torch.randn(self.freq_size_out))

        # 分类器额外的输出头
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.classifier = nn.Linear(configs.enc_in * self.output_len, configs.num_class)

    # Fre-domain Linear Layer
    # 输入频域上的复数嵌入(real,imag), 经过一个复数线性变换层
    # new_real = real*r - imag*i + rb
    # new_imag = imag*r + real*i + ib
    # 随后基于(new_real,new_imag)重新组合得到复数函数值, 这作为Fre-domain Linear Layer的输出结果
    def FreMLP(self, x, r, i, rb, ib):
        # print('############# FrsTS.FreMLP-1')
        # print(x.real.shape)     # torch.Size([32, 7, 169])
        # print(x.imag.shape)     # torch.Size([32, 7, 169])
        # print(r.shape)          # torch.Size([169, 49])
        # print(i.shape)          # torch.Size([169, 49])
        # print(rb.shape)         # torch.Size([49])
        # print(ib.shape)         # torch.Size([49])
        o1_real = F.relu(torch.einsum('bcl,lf->bcf', x.real, r) - torch.einsum('bcl,lf->bcf', x.imag, i) + rb)
        o1_imag = F.relu(torch.einsum('bcl,lf->bcf', x.imag, r) + torch.einsum('bcl,lf->bcf', x.real, i) + ib)
        # print(o1_real.shape)    # torch.Size([32, 7, 49])
        # print(o1_imag.shape)    # torch.Size([32, 7, 49])

        y = torch.stack([o1_real, o1_imag], dim=-1)
        # print(y.shape)          # torch.Size([32, 7, 49, 2])
        # print(self.sparsity_threshold)  # 0.01
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        # print(y.shape)          # torch.Size([32, 7, 49, 2])
        y = torch.view_as_complex(y)
        # print(y.shape)          # torch.Size([32, 7, 49])
        # print('############# FrsTS.FreMLP-2')
        return y

    # Frequency Sequence Learner
    # 输入时域上的观测序列temp_in(32,7,336)
    # FFT将temp_in转换为频域上的观测序列freq_in(32,7,169)
    # 通过频域上的复数线性变换层, 将freq_in(32,7,169)转换为freq_out(32,7,49)
    # iFFT将freq_out转换为时域上的预测序列freq_out(32,7,96)
    # 输出时域上的预测序列temp_out(32,7,96)
    def MLP_temporal(self, x):
        # print('############# FrsTS.MLP_temporal-1')
        # FFT
        # print(x.shape)          # torch.Size([32, 7, 336])
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        # print(x.shape)          # torch.Size([32, 7, 169])

        # Frequency MLP
        # print(x.shape)          # torch.Size([32, 7, 169])
        # print(self.r2.shape)    # torch.Size([169, 49])
        # print(self.i2.shape)    # torch.Size([169, 49])
        # print(self.rb2.shape)   # torch.Size([49])
        # print(self.ib2.shape)   # torch.Size([49])
        y = self.FreMLP(x, self.r, self.i, self.rb, self.ib)
        # print(y.shape)          # torch.Size([32, 7, 49])

        # iFFT
        # print(self.output_len)  # 96
        x = torch.fft.irfft(y, n=self.output_len, dim=2, norm="ortho")
        # print(x.shape)          # torch.Size([32, 7, 96])
        # print('############# FrsTS.MLP_temporal-2')
        return x

    def forecast(self, x_enc):
        # print('############# FrsTS.forecast-1')
        x_enc = self.normalize_layer(x_enc, 'norm')
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x = x_enc.permute(0, 2, 1)
        # print(x.shape)                      # torch.Size([32, 7, 336])
        x = self.MLP_temporal(x)
        # print(x.shape)                      # torch.Size([32, 7, 96])
        x = x.permute(0, 2, 1)
        x = self.normalize_layer(x, 'denorm')
        # print(x.shape)                      # torch.Size([32, 96, 7])
        # print('############# FrsTS.forecast-2')
        return x

    def imputation(self, x_enc):
        # print('############# FrsTS.forecast-1')
        x_enc = self.normalize_layer(x_enc, 'norm')
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x = x_enc.permute(0, 2, 1)
        # print(x.shape)                      # torch.Size([32, 7, 336])
        x = self.MLP_temporal(x)
        # print(x.shape)                      # torch.Size([32, 7, 336])
        x = x.permute(0, 2, 1)
        x = self.normalize_layer(x, 'denorm')
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print('############# FrsTS.forecast-2')
        return x

    def anomaly_detection(self, x_enc):
        # print('############# FrsTS.forecast-1')
        x_enc = self.normalize_layer(x_enc, 'norm')
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x = x_enc.permute(0, 2, 1)
        # print(x.shape)                      # torch.Size([32, 7, 336])
        x = self.MLP_temporal(x)
        # print(x.shape)                      # torch.Size([32, 7, 336])
        x = x.permute(0, 2, 1)
        x = self.normalize_layer(x, 'denorm')
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print('############# FrsTS.forecast-2')
        return x

    def classification(self, x_enc):
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        x = x_enc.permute(0, 2, 1)
        # print(x.shape)                      # torch.Size([32, 7, 336])
        x = self.MLP_temporal(x)
        # print(x.shape)                      # torch.Size([32, 7, 336])
        x = x.permute(0, 2, 1)

        # Output
        x = self.act(x)
        x = self.dropout(x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        x = x.reshape(x.shape[0], -1).contiguous()
        # print(x.shape)                      # torch.Size([32, 2352])
        x = self.classifier(x)
        # print(x.shape)                      # torch.Size([32, 4])
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out
        return None

    def my_show_weights_(self, save_path):
        import pandas as pd
        import matplotlib.pyplot as plt
        r = np.abs(self.r.detach().numpy())
        rb = np.abs(self.rb.detach().numpy())
        i = np.abs(self.i.detach().numpy())
        ib = np.abs(self.ib.detach().numpy())

        pd.plotting.register_matplotlib_converters()
        plt.rc("figure", figsize=(6, 8))
        plt.rc("font", size=10)

        plt.subplot(4, 1, 1)
        plt.grid(True)
        plt.plot(np.arange(1, len(rb) + 1), rb)
        plt.ylabel('Real basis')

        plt.subplot(4, 1, 2)
        plt.imshow(r, aspect='auto')   # plt.colorbar()
        plt.ylabel('Real weights')

        plt.subplot(4, 1, 3)
        plt.grid(True)
        plt.plot(np.arange(1, len(ib) + 1), ib)
        plt.ylabel('Imag basis')

        plt.subplot(4, 1, 4)
        plt.imshow(i, aspect='auto')   # plt.colorbar()
        plt.ylabel('Imag weights')

        np.save(f'{save_path}_r.npy', self.r.detach().numpy())
        np.save(f'{save_path}_rb.npy', self.rb.detach().numpy())
        np.save(f'{save_path}_i.npy', self.i.detach().numpy())
        np.save(f'{save_path}_ib.npy', self.ib.detach().numpy())

        # plt.show()
        plt.savefig(f'{save_path}.png')
        plt.savefig(f'{save_path}.svg')
        plt.close()
        return None

    def my_show_weights(self, save_path):
        import pandas as pd
        import matplotlib.pyplot as plt
        r = np.abs(self.r.detach().numpy())
        rb = np.abs(self.rb.detach().numpy())
        i = np.abs(self.i.detach().numpy())
        ib = np.abs(self.ib.detach().numpy())

        plt.imshow(r, aspect='auto')
        plt.colorbar()
        plt.savefig(f'{save_path}_r.svg')
        plt.close()

        plt.imshow(i, aspect='auto')
        plt.colorbar()
        plt.savefig(f'{save_path}_i.svg')
        plt.close()
        return None


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


class FullAttention_MultiHead(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention_MultiHead, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        from math import sqrt
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A


class Model_v21(nn.Module):
    def __init__(self, configs, is_show=False):
        super(Model_v21, self).__init__()
        self.task_name = configs.task_name
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)

        # Weights of AttnLayer
        self.query_projection = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.key_projection = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.value_projection = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.out_projection = nn.Linear(3*configs.enc_in, configs.enc_in)
        self.attention = FullAttention()

        # Weights of FFNLayer
        self.conv1 = nn.Conv1d(in_channels=configs.enc_in, out_channels=3*configs.enc_in, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=3*configs.enc_in, out_channels=configs.enc_in, kernel_size=1)
        self.norm1 = nn.LayerNorm(configs.enc_in)
        self.norm2 = nn.LayerNorm(configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # 分类器额外的输出头
        self.seq_len = configs.seq_len
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.classifier = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

        # 实验分为两部分, 第一部分旨在展示"未解耦情况下观测序列中任意两个timestep之间的attn",
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 旨在计算保存每个batch中输入的x(32,336,7)和attn(32,336,336)
        # 第二部分旨在展示"解耦之后每个波动序列中任意两个timestep之间的attn", 此时直接加载步骤1中的
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 直接加载步骤1中的x(32,336,7), 将x解耦为多个波动序列, 计算并保存每个波动序列对应的attn(32,336,336)
        self.is_show = is_show
        self.save_path = f'./show_{configs.model}_v2/{configs.task_name}_{configs.model_id}'

    def save_attention(self, attn_observation, x_enc):
        # 获取self.save_path目录下全部文件的名字
        import os
        npy_list = os.listdir(f'{self.save_path}/npy')
        npy_list_observation = [npy_str for npy_str in npy_list if 'observation' in npy_str]
        npy_list_pattern = [npy_str for npy_str in npy_list if 'pattern' in npy_str]
        npy_list_x = [npy_str for npy_str in npy_list if 'x' in npy_str]
        assert len(npy_list) == len(npy_list_observation) + len(npy_list_pattern) + len(npy_list_x)

        # 保存npy文件
        np.save(f'{self.save_path}/npy/{len(npy_list_observation)}_attn_observation.npy', attn_observation.cpu().detach().numpy())
        np.save(f'{self.save_path}/npy/{len(npy_list_observation)}_x_enc.npy', x_enc.cpu().detach().numpy())

    def encode(self, x):
        # print('############# ShowModel.encode-1')
        # AttentionLayer
        # print(x.shape)                      # torch.Size([32, 336, 7])
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
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
        new_x = self.out_projection(new_x)
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = x + self.dropout(new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])

        # FFNLayer
        x = self.norm1(x)
        new_x = x
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        new_x = self.dropout(self.activation(self.conv1(new_x.transpose(-1, 1))))
        # print(new_x.shape)                  # torch.Size([32, 21, 336])
        new_x = self.dropout(self.conv2(new_x).transpose(-1, 1))
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = self.norm2(x + new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print('############# ShowModel.encode-2')
        return x, attn

    def forecast(self, x_enc):
        # print('############# ShowModel.forecast-1')
        x = copy.deepcopy(x_enc)
        x = self.normalize_layer(x, 'norm')
        # print(x.shape)                      # torch.Size([32, 336, 7])
        x, attn = self.encode(x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print(attn.shape)                   # torch.Size([32, 336, 336])
        x = self.projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        # print(x.shape)                      # torch.Size([32, 96, 7])
        x = self.normalize_layer(x, 'denorm')
        # print(x.shape)                      # torch.Size([32, 96, 7])

        if self.is_show:
            self.save_attention(attn_observation=attn, x_enc=x_enc)

        # print('############# ShowModel.forecast-2')
        return x

    def imputation(self, x_enc):
        # print('############# ShowModel.imputation-1')
        x = copy.deepcopy(x_enc)
        x = self.normalize_layer(x, 'norm')
        # print(x.shape)                      # torch.Size([32, 336, 7])
        x, attn = self.encode(x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print(attn.shape)                   # torch.Size([32, 336, 336])
        x = self.normalize_layer(x, 'denorm')
        # print(x.shape)                      # torch.Size([32, 336, 7])

        if self.is_show:
            self.save_attention(attn_observation=attn, x_enc=x_enc)

        # print('############# ShowModel.imputation-2')
        return x

    def anomaly_detection(self, x_enc):
        # print('############# ShowModel.anomaly-1')
        x = copy.deepcopy(x_enc)
        x = self.normalize_layer(x, 'norm')
        # print(x.shape)                      # torch.Size([32, 336, 7])
        x, attn = self.encode(x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print(attn.shape)                   # torch.Size([32, 336, 336])
        x = self.normalize_layer(x, 'denorm')
        # print(x.shape)                      # torch.Size([32, 336, 7])

        if self.is_show:
            self.save_attention(attn_observation=attn, x_enc=x_enc)

        # print('############# ShowModel.anomaly-2')
        return x

    def classification(self, x_enc):
        # print('############# ShowModel.anomaly-1')
        x = copy.deepcopy(x_enc)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        x, attn = self.encode(x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        # print(attn.shape)                   # torch.Size([32, 336, 336])
        # print('############# ShowModel.anomaly-2')

        # Output
        x = self.act(x)
        x = self.dropout(x)
        # print(x.shape)                      # torch.Size([32, 336, 7])
        x = x.reshape(x.shape[0], -1).contiguous()
        # print(x.shape)                      # torch.Size([32, 2352])
        x = self.classifier(x)
        # print(x.shape)                      # torch.Size([32, 4])

        if self.is_show:
            self.save_attention(attn_observation=attn, x_enc=x_enc)

        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out


class Model_v22(nn.Module):
    def __init__(self, configs, is_show=False):
        super(Model_v22, self).__init__()
        self.task_name = configs.task_name
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)

        # Pattern:1
        # Weights of AttnLayer
        self.query_projection_p1 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.key_projection_p1 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.value_projection_p1 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.out_projection_p1 = nn.Linear(3*configs.enc_in, configs.enc_in)
        self.attention = FullAttention()
        # Weights of FFNLayer
        self.conv1_p1 = nn.Conv1d(in_channels=configs.enc_in, out_channels=3*configs.enc_in, kernel_size=1)
        self.conv2_p1 = nn.Conv1d(in_channels=3*configs.enc_in, out_channels=configs.enc_in, kernel_size=1)
        self.norm1_p1 = nn.LayerNorm(configs.enc_in)
        self.norm2_p1 = nn.LayerNorm(configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:2
        # Weights of AttnLayer
        self.query_projection_p2 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.key_projection_p2 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.value_projection_p2 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.out_projection_p2 = nn.Linear(3*configs.enc_in, configs.enc_in)
        self.attention = FullAttention()
        # Weights of FFNLayer
        self.conv1_p2 = nn.Conv1d(in_channels=configs.enc_in, out_channels=3*configs.enc_in, kernel_size=1)
        self.conv2_p2 = nn.Conv1d(in_channels=3*configs.enc_in, out_channels=configs.enc_in, kernel_size=1)
        self.norm1_p2 = nn.LayerNorm(configs.enc_in)
        self.norm2_p2 = nn.LayerNorm(configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:3
        # Weights of AttnLayer
        self.query_projection_p3 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.key_projection_p3 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.value_projection_p3 = nn.Linear(configs.enc_in, 3*configs.enc_in)
        self.out_projection_p3 = nn.Linear(3*configs.enc_in, configs.enc_in)
        self.attention = FullAttention()
        # Weights of FFNLayer
        self.conv1_p3 = nn.Conv1d(in_channels=configs.enc_in, out_channels=3*configs.enc_in, kernel_size=1)
        self.conv2_p3 = nn.Conv1d(in_channels=3*configs.enc_in, out_channels=configs.enc_in, kernel_size=1)
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

    def my_clean(self, array_2d):
        # orig_shape = array_2d.shape
        # array_1d = array_2d.flatten().reshape(1, -1)

        # 前99.5%的值不超过0.026, 然而最大的一批值为0.16,
        # 因此我们的处理思路就是将前0.5%,1%,1.5%,2%,2.5%这五个梯度的值替换为固定的且不离群的值,
        # 从而避免整个attn中只有少部分值较大的情况,
        min_ = np.min(array_2d)
        max_ = np.max(array_2d)
        # print(min_)     # 5.921301e-09
        # print(max_)     # 0.1620357
        p1 = np.percentile(array_2d, 95.0)
        p2 = np.percentile(array_2d, 96.0)
        p3 = np.percentile(array_2d, 97.0)
        p4 = np.percentile(array_2d, 98.0)
        p5 = np.percentile(array_2d, 99.0)
        # print(p1)       # 0.01196886959951371
        # print(p2)       # 0.013415353000164002
        # print(p3)       # 0.015470577869564293
        # print(p4)       # 0.018583029881119824
        # print(p5)       # 0.026033291639760116

        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                temp = array_2d[i, j]
                if temp > p5:
                    array_2d[i, j] = 2 * p5
                elif temp > p4:
                    array_2d[i, j] = p5
                elif temp > p3:
                    array_2d[i, j] = p4
                elif temp > p2:
                    array_2d[i, j] = p3
                elif temp > p1:
                    array_2d[i, j] = p2
                # elif temp < p2:
                #     array_2d[i, j] += 2 * diss
                # elif temp < p3:
                #     array_2d[i, j] += 3 * diss
                # elif temp < p4:
                #     array_2d[i, j] += 4 * diss
                # elif temp < p5:
                #     array_2d[i, j] += 5 * diss
                # elif temp < p6:
                #     array_2d[i, j] += 6 * diss
                # elif temp < p7:
                #     array_2d[i, j] += 7 * diss
                # else:
                #     array_2d[i, j] += 8 * diss

        # array_2d = array_1d.reshape(orig_shape)
        return array_2d

    def plot_attention(self):
        # 获取self.save_path目录下全部文件的名字
        import os
        npy_list = os.listdir(f'{self.save_path}/npy')
        npy_list_observation = [npy_str for npy_str in npy_list if 'observation' in npy_str]
        npy_list_pattern = [npy_str for npy_str in npy_list if 'pattern' in npy_str]
        npy_list_x = [npy_str for npy_str in npy_list if 'x' in npy_str]
        assert len(npy_list_observation) == len(npy_list_x)
        assert len(npy_list_observation) * 3 == len(npy_list_pattern)
        assert len(npy_list) == len(npy_list_observation) + len(npy_list_pattern) + len(npy_list_x)

        print(f'Start Plot: {self.save_path}, i: {len(npy_list_x)}, j: 256')
        for idx in range(len(npy_list_x)):
            # 加载数据
            attn_observation = np.load(f'{self.save_path}/npy/{idx}_attn_observation.npy')
            attn_pattern1 = np.load(f'{self.save_path}/npy/{idx}_attn_pattern1.npy')
            attn_pattern2 = np.load(f'{self.save_path}/npy/{idx}_attn_pattern2.npy')
            attn_pattern3 = np.load(f'{self.save_path}/npy/{idx}_attn_pattern3.npy')
            # print(attn_observation.shape)   # (256, 336, 336)
            # print(attn_pattern1.shape)      # (256, 336, 336)
            # print(attn_pattern2.shape)      # (256, 336, 336)
            # print(attn_pattern3.shape)      # (256, 336, 336)

            # 开始画图
            idx1 = attn_observation.shape[0] // 4
            idx2 = idx1 * 2
            idx3 = idx1 * 3
            os.makedirs(f'{self.save_path}/image', exist_ok=True)
            for jdx in [idx1, idx2, idx3]:
                attn_observation_ = attn_observation[jdx, :, :]
                attn_pattern1_ = attn_pattern1[jdx, :, :]
                attn_pattern2_ = attn_pattern2[jdx, :, :]
                attn_pattern3_ = attn_pattern3[jdx, :, :]

                # 归一化清洗
                attn_observation_ = self.my_clean(attn_observation_)
                attn_pattern1_ = self.my_clean(attn_pattern1_)
                attn_pattern2_ = self.my_clean(attn_pattern2_)
                attn_pattern3_ = self.my_clean(attn_pattern3_)

                import pandas as pd
                import matplotlib.pyplot as plt
                pd.plotting.register_matplotlib_converters()
                plt.rc("figure", figsize=(6, 8))
                plt.rc("font", size=10)

                plt.subplot(4, 1, 1)
                plt.imshow(attn_observation_, aspect='auto')    # plt.colorbar()
                plt.ylabel('Observation')

                plt.subplot(4, 1, 2)
                plt.imshow(attn_pattern1_, aspect='auto')       # plt.colorbar()
                plt.ylabel('Pattern1')

                plt.subplot(4, 1, 3)
                plt.imshow(attn_pattern2_, aspect='auto')       # plt.colorbar()
                plt.ylabel('Pattern2')

                plt.subplot(4, 1, 4)
                plt.imshow(attn_pattern3_, aspect='auto')       # plt.colorbar()
                plt.ylabel('Pattern3')

                # 保存图片
                # plt.show()
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.png')
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.svg')
                plt.close()

        return None

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

    def forward_myself(self, device):
        import os
        npy_list = os.listdir(f'{self.save_path}/npy')
        npy_list_x = [npy_str for npy_str in npy_list if 'x' in npy_str]
        for idx in range(len(npy_list_x)):
            x_enc = np.load(f'{self.save_path}/npy/{idx}_x_enc.npy')
            x_enc = torch.tensor(x_enc).float().to(device)
            self.forward(x_enc=x_enc, idx=idx)


class Model_v3(nn.Module):
    def __init__(self, configs, is_show=False):
        super(Model_v3, self).__init__()
        self.task_name = configs.task_name
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)

        self.patch_len = 8

        # Weights of AttnLayer
        self.query_projection = nn.Linear(self.patch_len, 3*self.patch_len)
        self.key_projection = nn.Linear(self.patch_len, 3*self.patch_len)
        self.value_projection = nn.Linear(self.patch_len, 3*self.patch_len)
        self.out_projection = nn.Linear(3*self.patch_len, self.patch_len)
        self.attention = FullAttention()
        # Weights of FFNLayer
        self.conv1 = nn.Conv1d(in_channels=self.patch_len, out_channels=3*self.patch_len, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=3*self.patch_len, out_channels=self.patch_len, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.patch_len)
        self.norm2 = nn.LayerNorm(self.patch_len)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:Trend and Projection:1,2,3
        in_len = math.ceil(configs.seq_len/self.patch_len) * self.patch_len * 3
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(in_len, configs.pred_len, bias=True)
            self.projection.apply(self._init_weights)
            self.up_ct = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(in_len, configs.seq_len, bias=True)
            self.projection.apply(self._init_weights)
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'classification':
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.classifier = nn.Linear(configs.enc_in * (configs.seq_len + in_len), configs.num_class)
            self.classifier.apply(self._init_weights)

        # 实验分为两部分, 第一部分旨在展示"未解耦情况下观测序列中任意两个timestep之间的attn",
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 旨在计算保存每个batch中输入的x(32,336,7)和attn(32,336,336)
        # 第二部分旨在展示"解耦之后每个波动序列中任意两个timestep之间的attn", 此时直接加载步骤1中的
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 直接加载步骤1中的x(32,336,7), 将x解耦为多个波动序列, 计算并保存每个波动序列对应的attn(32,336,336)
        self.is_show = is_show
        self.save_path = f'./show_{configs.model}_v3/{configs.task_name}_{configs.model_id}'

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

    def my_clean(self, array_2d):
        # orig_shape = array_2d.shape
        # array_1d = array_2d.flatten().reshape(1, -1)

        # 前99.5%的值不超过0.026, 然而最大的一批值为0.16,
        # 因此我们的处理思路就是将前0.5%,1%,1.5%,2%,2.5%这五个梯度的值替换为固定的且不离群的值,
        # 从而避免整个attn中只有少部分值较大的情况,
        min_ = np.min(array_2d)
        max_ = np.max(array_2d)
        # print(min_)     # 5.921301e-09
        # print(max_)     # 0.1620357
        p1 = np.percentile(array_2d, 95.0)
        p2 = np.percentile(array_2d, 96.0)
        p3 = np.percentile(array_2d, 97.0)
        p4 = np.percentile(array_2d, 98.0)
        p5 = np.percentile(array_2d, 99.0)
        # print(p1)       # 0.01196886959951371
        # print(p2)       # 0.013415353000164002
        # print(p3)       # 0.015470577869564293
        # print(p4)       # 0.018583029881119824
        # print(p5)       # 0.026033291639760116

        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                temp = array_2d[i, j]
                if temp > p5:
                    array_2d[i, j] = 2 * p5
                elif temp > p4:
                    array_2d[i, j] = p5
                elif temp > p3:
                    array_2d[i, j] = p4
                elif temp > p2:
                    array_2d[i, j] = p3
                elif temp > p1:
                    array_2d[i, j] = p2
                # elif temp < p2:
                #     array_2d[i, j] += 2 * diss
                # elif temp < p3:
                #     array_2d[i, j] += 3 * diss
                # elif temp < p4:
                #     array_2d[i, j] += 4 * diss
                # elif temp < p5:
                #     array_2d[i, j] += 5 * diss
                # elif temp < p6:
                #     array_2d[i, j] += 6 * diss
                # elif temp < p7:
                #     array_2d[i, j] += 7 * diss
                # else:
                #     array_2d[i, j] += 8 * diss

        # array_2d = array_1d.reshape(orig_shape)
        return array_2d

    def plot_attention(self):

        # 获取self.save_path目录下全部文件的名字
        import os
        npy_list = os.listdir(f'{self.save_path}/npy')
        npy_list_attn = [npy_str for npy_str in npy_list if 'attn' in npy_str]
        assert len(npy_list) == len(npy_list_attn)

        print(f'Start Plot: {self.save_path}, i: {len(npy_list_attn)}, j: 256')
        for idx in range(len(npy_list_attn)):
            # 加载数据
            attn_attn = np.load(f'{self.save_path}/npy/{idx}_attn.npy')
            # print(attn_attn.shape)          # (256, 336, 336)

            # 开始画图
            idx1 = attn_attn.shape[0] // 4
            idx2 = idx1 * 2
            idx3 = idx1 * 3
            os.makedirs(f'{self.save_path}/image', exist_ok=True)
            for jdx in [idx1, idx2, idx3]:
                attn_attn_ = attn_attn[jdx, :, :]

                # 归一化清洗
                attn_attn_ = self.my_clean(attn_attn_)

                import matplotlib.pyplot as plt
                plt.imshow(attn_attn_, aspect='auto')    # plt.colorbar()

                # 保存图片
                # plt.show()
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.png')
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.svg')
                plt.close()

        return None

    def encode(self, x):
        # print('############# ShowModel.encode-1')
        # AttentionLayer
        # print(x.shape)                      # torch.Size([32, 336, 7])
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
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
        new_x = self.out_projection(new_x)
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = x + self.dropout(new_x)
        # print(x.shape)                      # torch.Size([32, 336, 7])

        # FFNLayer
        x = self.norm1(x)
        new_x = x
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        new_x = self.dropout(self.activation(self.conv1(new_x.transpose(-1, 1))))
        # print(new_x.shape)                  # torch.Size([32, 21, 336])
        new_x = self.dropout(self.conv2(new_x).transpose(-1, 1))
        # print(new_x.shape)                  # torch.Size([32, 336, 7])
        x = self.norm2(x + new_x)
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2]//self.patch_len+1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        # print(self.patch_len)       # 8
        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # print(c1.shape)             # torch.Size([32, 7, 42, 8])
        # print(c2.shape)             # torch.Size([32, 7, 42, 8])
        # print(c3.shape)             # torch.Size([32, 7, 42, 8])
        c1 = c1.reshape(-1, c1.shape[2], c1.shape[3]).contiguous()
        c2 = c2.reshape(-1, c2.shape[2], c2.shape[3]).contiguous()
        c3 = c3.reshape(-1, c3.shape[2], c3.shape[3]).contiguous()
        # print(c1.shape)             # torch.Size([224, 42, 8])
        # print(c2.shape)             # torch.Size([224, 42, 8])
        # print(c3.shape)             # torch.Size([224, 42, 8])
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([224, 126, 8])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([224, 126, 8])
        # print(attn.shape)           # torch.Size([224, 126, 126])

        # print(c.shape)              # torch.Size([224, 126, 8])
        c = c.reshape(c.shape[0], -1).contiguous()
        # print(c.shape)              # torch.Size([224, 1008])
        o = self.projection(c)
        # print(o.shape)              # torch.Size([224, 96])
        o = o.reshape(x.shape[0], x.shape[1], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 7, 96])
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 96, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 96])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 96, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 96, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.reshape(-1, c1.shape[2], c1.shape[3]).contiguous()
        c2 = c2.reshape(-1, c2.shape[2], c2.shape[3]).contiguous()
        c3 = c3.reshape(-1, c3.shape[2], c3.shape[3]).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c1.shape)             # torch.Size([224, 42, 8])
        # print(c2.shape)             # torch.Size([224, 42, 8])
        # print(c3.shape)             # torch.Size([224, 42, 8])
        # print(c.shape)              # torch.Size([224, 126, 8])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([224, 126, 8])
        # print(attn.shape)           # torch.Size([224, 126, 126])
        c = c.reshape(c.shape[0], -1).contiguous()
        # print(c.shape)              # torch.Size([224, 1008])
        o = self.projection(c)
        # print(o.shape)              # torch.Size([224, 336])
        o = o.reshape(x.shape[0], x.shape[1], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 7, 336])
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 336, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 336, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 336, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.reshape(-1, c1.shape[2], c1.shape[3]).contiguous()
        c2 = c2.reshape(-1, c2.shape[2], c2.shape[3]).contiguous()
        c3 = c3.reshape(-1, c3.shape[2], c3.shape[3]).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c1.shape)             # torch.Size([224, 42, 8])
        # print(c2.shape)             # torch.Size([224, 42, 8])
        # print(c3.shape)             # torch.Size([224, 42, 8])
        # print(c.shape)              # torch.Size([224, 126, 8])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([224, 126, 8])
        # print(attn.shape)           # torch.Size([224, 126, 126])
        c = c.reshape(c.shape[0], -1).contiguous()
        # print(c.shape)              # torch.Size([224, 1008])
        o = self.projection(c)
        # print(o.shape)              # torch.Size([224, 336])
        o = o.reshape(x.shape[0], x.shape[1], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 7, 336])
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 336, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 336, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 336, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.reshape(-1, c1.shape[2], c1.shape[3]).contiguous()
        c2 = c2.reshape(-1, c2.shape[2], c2.shape[3]).contiguous()
        c3 = c3.reshape(-1, c3.shape[2], c3.shape[3]).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c1.shape)             # torch.Size([224, 42, 8])
        # print(c2.shape)             # torch.Size([224, 42, 8])
        # print(c3.shape)             # torch.Size([224, 42, 8])
        # print(c.shape)              # torch.Size([224, 126, 8])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([224, 126, 8])
        # print(attn.shape)           # torch.Size([224, 126, 126])
        c = c.reshape(c.shape[0], -1).contiguous()
        # print(c.shape)              # torch.Size([224, 1008])
        o = c.reshape(x.shape[0], x.shape[1], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 7, 1008])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])

        o = torch.cat([ot, o], 2)
        # print(o.shape)              # torch.Size([32, 7, 1344])

        # Output
        o = self.act(o)
        o = self.dropout(o)
        # print(o.shape)              # torch.Size([32, 7, 1344])
        o = o.reshape(o.shape[0], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 9408])
        o = self.classifier(o)
        # print(o.shape)              # torch.Size([32, 4])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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


class Model_v4(nn.Module):
    def __init__(self, configs, is_show=False):
        super(Model_v4, self).__init__()
        self.task_name = configs.task_name
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)

        self.patch_len = 8
        self.n_head = 8

        # Weights of AttnLayer
        self.query_projection = nn.Linear(self.patch_len*configs.enc_in, 3*self.patch_len*configs.enc_in)
        self.key_projection = nn.Linear(self.patch_len*configs.enc_in, 3*self.patch_len*configs.enc_in)
        self.value_projection = nn.Linear(self.patch_len*configs.enc_in, 3*self.patch_len*configs.enc_in)
        self.out_projection = nn.Linear(3*self.patch_len*configs.enc_in, self.patch_len*configs.enc_in)
        self.attention = FullAttention_MultiHead()
        # Weights of FFNLayer
        self.conv1 = nn.Conv1d(in_channels=self.patch_len*configs.enc_in, out_channels=3*self.patch_len*configs.enc_in, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=3*self.patch_len*configs.enc_in, out_channels=self.patch_len*configs.enc_in, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.patch_len*configs.enc_in)
        self.norm2 = nn.LayerNorm(self.patch_len*configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:Trend and Projection:1,2,3
        in_len = math.ceil(configs.seq_len/self.patch_len) * self.patch_len * 3
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(in_len, configs.pred_len, bias=True)
            self.projection.apply(self._init_weights)
            self.up_ct = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(in_len, configs.seq_len, bias=True)
            self.projection.apply(self._init_weights)
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'classification':
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.classifier = nn.Linear(configs.enc_in * (configs.seq_len + in_len), configs.num_class)
            self.classifier.apply(self._init_weights)

        # 实验分为两部分, 第一部分旨在展示"未解耦情况下观测序列中任意两个timestep之间的attn",
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 旨在计算保存每个batch中输入的x(32,336,7)和attn(32,336,336)
        # 第二部分旨在展示"解耦之后每个波动序列中任意两个timestep之间的attn", 此时直接加载步骤1中的
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 直接加载步骤1中的x(32,336,7), 将x解耦为多个波动序列, 计算并保存每个波动序列对应的attn(32,336,336)
        self.is_show = is_show
        self.save_path = f'./show_{configs.model}_v4/{configs.task_name}_{configs.model_id}'

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

    def my_clean(self, array_2d):
        # orig_shape = array_2d.shape
        # array_1d = array_2d.flatten().reshape(1, -1)

        # 前99.5%的值不超过0.026, 然而最大的一批值为0.16,
        # 因此我们的处理思路就是将前0.5%,1%,1.5%,2%,2.5%这五个梯度的值替换为固定的且不离群的值,
        # 从而避免整个attn中只有少部分值较大的情况,
        min_ = np.min(array_2d)
        max_ = np.max(array_2d)
        # print(min_)     # 5.921301e-09
        # print(max_)     # 0.1620357
        p1 = np.percentile(array_2d, 95.0)
        p2 = np.percentile(array_2d, 96.0)
        p3 = np.percentile(array_2d, 97.0)
        p4 = np.percentile(array_2d, 98.0)
        p5 = np.percentile(array_2d, 99.0)
        # print(p1)       # 0.01196886959951371
        # print(p2)       # 0.013415353000164002
        # print(p3)       # 0.015470577869564293
        # print(p4)       # 0.018583029881119824
        # print(p5)       # 0.026033291639760116

        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                temp = array_2d[i, j]
                if temp > p5:
                    array_2d[i, j] = 2 * p5
                elif temp > p4:
                    array_2d[i, j] = p5
                elif temp > p3:
                    array_2d[i, j] = p4
                elif temp > p2:
                    array_2d[i, j] = p3
                elif temp > p1:
                    array_2d[i, j] = p2
                # elif temp < p2:
                #     array_2d[i, j] += 2 * diss
                # elif temp < p3:
                #     array_2d[i, j] += 3 * diss
                # elif temp < p4:
                #     array_2d[i, j] += 4 * diss
                # elif temp < p5:
                #     array_2d[i, j] += 5 * diss
                # elif temp < p6:
                #     array_2d[i, j] += 6 * diss
                # elif temp < p7:
                #     array_2d[i, j] += 7 * diss
                # else:
                #     array_2d[i, j] += 8 * diss

        # array_2d = array_1d.reshape(orig_shape)
        return array_2d

    def plot_attention(self):
        # 获取self.save_path目录下全部文件的名字
        import os
        npy_list = os.listdir(f'{self.save_path}/npy')
        npy_list_attn = [npy_str for npy_str in npy_list if 'attn' in npy_str]
        assert len(npy_list) == len(npy_list_attn)

        print(f'Start Plot: {self.save_path}, i: {len(npy_list_attn)}, j: 256')
        for idx in range(len(npy_list_attn)):
            # 加载数据
            attn_attn = np.load(f'{self.save_path}/npy/{idx}_attn.npy')
            # print(attn_attn.shape)          # (256, 8, 126, 126)

            # 开始画图
            os.makedirs(f'{self.save_path}/image', exist_ok=True)
            for jdx in range(attn_attn.shape[1]):
                attn_attn_ = attn_attn[0, jdx, :, :]
                # print(attn_attn_.shape)     # (126, 126)

                # 归一化清洗
                attn_attn_ = self.my_clean(attn_attn_)

                import matplotlib.pyplot as plt
                plt.imshow(attn_attn_, aspect='auto')    # plt.colorbar()

                # 保存图片
                # plt.show()
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.png')
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.svg')
                plt.close()

        return None

    def encode(self, x):
        # print('############# ShowModel.encode-1')
        B, N, D = x.shape

        # AttentionLayer
        # print(x.shape)                      # torch.Size([32, 126, 56])
        query = self.query_projection(x).view(B, N, self.n_head, -1)
        key = self.key_projection(x).view(B, N, self.n_head, -1)
        value = self.value_projection(x).view(B, N, self.n_head, -1)
        # print(query.shape)                  # torch.Size([32, 126, 8, 21])
        # print(key.shape)                    # torch.Size([32, 126, 8, 21])
        # print(value.shape)                  # torch.Size([32, 126, 8, 21])
        new_x, attn = self.attention(
            query,
            key,
            value,
        )
        # print(new_x.shape)                  # torch.Size([32, 126, 8, 21])
        new_x = new_x.view(B, N, -1)
        # print(new_x.shape)                  # torch.Size([32, 126, 168])
        new_x = self.out_projection(new_x)
        # print(new_x.shape)                  # torch.Size([32, 126, 56])
        x = x + self.dropout(new_x)
        # print(x.shape)                      # torch.Size([32, 126, 56])

        # print(attn.shape)                   # torch.Size([32, 336, 336])

        # FFNLayer
        x = self.norm1(x)
        new_x = x
        # print(new_x.shape)                  # torch.Size([32, 126, 56])
        new_x = self.dropout(self.activation(self.conv1(new_x.transpose(-1, 1))))
        # print(new_x.shape)                  # torch.Size([32, 126, 126])
        new_x = self.dropout(self.conv2(new_x).transpose(-1, 1))
        # print(new_x.shape)                  # torch.Size([32, 126, 56])
        x = self.norm2(x + new_x)
        # print(x.shape)                      # torch.Size([32, 126, 56])
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2]//self.patch_len+1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        # print(self.patch_len)       # 8
        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # print(c1.shape)             # torch.Size([32, 7, 42, 8])
        # print(c2.shape)             # torch.Size([32, 7, 42, 8])
        # print(c3.shape)             # torch.Size([32, 7, 42, 8])
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        # print(c1.shape)             # torch.Size([32, 42, 7, 8])
        # print(c2.shape)             # torch.Size([32, 42, 7, 8])
        # print(c3.shape)             # torch.Size([32, 42, 7, 8])
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        # print(c1.shape)             # torch.Size([32, 42, 56])
        # print(c2.shape)             # torch.Size([32, 42, 56])
        # print(c3.shape)             # torch.Size([32, 42, 56])
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([32, 126, 56])
        # print(attn.shape)           # torch.Size([32, 126, 126])

        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        # print(c.shape)              # torch.Size([32, 126, 7, 8])
        c = c.permute(0, 2, 1, 3).contiguous()
        # print(c.shape)              # torch.Size([32, 7, 126, 8])
        c = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        # print(c.shape)              # torch.Size([32, 7, 1008])
        o = self.projection(c)
        # print(o.shape)              # torch.Size([32, 7, 96])
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 96, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 96])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 96, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 96, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        c = c.permute(0, 2, 1, 3).contiguous()
        c = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        o = self.projection(c)
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 336, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 336, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 336, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        c = c.permute(0, 2, 1, 3).contiguous()
        c = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        o = self.projection(c)
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 336, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 336, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 336, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([32, 126, 56])
        # print(attn.shape)           # torch.Size([32, 126, 126])
        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        # print(c.shape)              # torch.Size([32, 126, 7, 8])
        c = c.permute(0, 2, 1, 3).contiguous()
        # print(c.shape)              # torch.Size([32, 7, 126, 8])
        o = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 7, 1008])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])

        o = torch.cat([ot, o], 2)
        # print(o.shape)              # torch.Size([32, 7, 1344])

        # Output
        o = self.act(o)
        o = self.dropout(o)
        # print(o.shape)              # torch.Size([32, 7, 1344])
        o = o.reshape(o.shape[0], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 9408])
        o = self.classifier(o)
        # print(o.shape)              # torch.Size([32, 4])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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


class Model(nn.Module):
    def __init__(self, configs, is_show=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)

        self.patch_len = 8
        self.n_head = 8

        # Weights of AttnLayer
        self.query_projection = nn.Linear(self.patch_len*configs.enc_in, 3*self.patch_len*configs.enc_in)
        self.key_projection = nn.Linear(self.patch_len*configs.enc_in, 3*self.patch_len*configs.enc_in)
        self.value_projection = nn.Linear(self.patch_len*configs.enc_in, 3*self.patch_len*configs.enc_in)
        self.out_projection = nn.Linear(3*self.patch_len*configs.enc_in, self.patch_len*configs.enc_in)
        self.attention = FullAttention_MultiHead()
        # Weights of FFNLayer
        self.conv1 = nn.Conv1d(in_channels=self.patch_len*configs.enc_in, out_channels=3*self.patch_len*configs.enc_in, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=3*self.patch_len*configs.enc_in, out_channels=self.patch_len*configs.enc_in, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.patch_len*configs.enc_in)
        self.norm2 = nn.LayerNorm(self.patch_len*configs.enc_in)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu

        # Pattern:Trend and Projection:1,2,3
        in_len = math.ceil(configs.seq_len/self.patch_len) * self.patch_len * 3
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(in_len, configs.pred_len, bias=True)
            self.projection.apply(self._init_weights)
            self.up_ct = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(in_len, configs.seq_len, bias=True)
            self.projection.apply(self._init_weights)
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
        elif self.task_name == 'classification':
            self.up_ct = nn.Linear(configs.seq_len, configs.seq_len, bias=True)
            self.up_ct.apply(self._init_weights)
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.classifier = nn.Linear(configs.enc_in * (configs.seq_len + in_len), configs.num_class)
            self.classifier.apply(self._init_weights)

        # 实验分为两部分, 第一部分旨在展示"未解耦情况下观测序列中任意两个timestep之间的attn",
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 旨在计算保存每个batch中输入的x(32,336,7)和attn(32,336,336)
        # 第二部分旨在展示"解耦之后每个波动序列中任意两个timestep之间的attn", 此时直接加载步骤1中的
        #   is_show在训练期间为false, 即正常训练
        #   is_show在推理期间为true, 直接加载步骤1中的x(32,336,7), 将x解耦为多个波动序列, 计算并保存每个波动序列对应的attn(32,336,336)
        self.is_show = is_show
        self.save_path = f'./show_{configs.model}_v4/{configs.task_name}_{configs.model_id}'

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

    def my_clean(self, array_2d):
        # orig_shape = array_2d.shape
        # array_1d = array_2d.flatten().reshape(1, -1)

        # 前99.5%的值不超过0.026, 然而最大的一批值为0.16,
        # 因此我们的处理思路就是将前0.5%,1%,1.5%,2%,2.5%这五个梯度的值替换为固定的且不离群的值,
        # 从而避免整个attn中只有少部分值较大的情况,
        min_ = np.min(array_2d)
        max_ = np.max(array_2d)
        # print(min_)     # 5.921301e-09
        # print(max_)     # 0.1620357
        p1 = np.percentile(array_2d, 95.0)
        p2 = np.percentile(array_2d, 96.0)
        p3 = np.percentile(array_2d, 97.0)
        p4 = np.percentile(array_2d, 98.0)
        p5 = np.percentile(array_2d, 99.0)
        # print(p1)       # 0.01196886959951371
        # print(p2)       # 0.013415353000164002
        # print(p3)       # 0.015470577869564293
        # print(p4)       # 0.018583029881119824
        # print(p5)       # 0.026033291639760116

        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                temp = array_2d[i, j]
                if temp > p5:
                    array_2d[i, j] = 2 * p5
                elif temp > p4:
                    array_2d[i, j] = p5
                elif temp > p3:
                    array_2d[i, j] = p4
                elif temp > p2:
                    array_2d[i, j] = p3
                elif temp > p1:
                    array_2d[i, j] = p2
                # elif temp < p2:
                #     array_2d[i, j] += 2 * diss
                # elif temp < p3:
                #     array_2d[i, j] += 3 * diss
                # elif temp < p4:
                #     array_2d[i, j] += 4 * diss
                # elif temp < p5:
                #     array_2d[i, j] += 5 * diss
                # elif temp < p6:
                #     array_2d[i, j] += 6 * diss
                # elif temp < p7:
                #     array_2d[i, j] += 7 * diss
                # else:
                #     array_2d[i, j] += 8 * diss

        # array_2d = array_1d.reshape(orig_shape)
        return array_2d

    def plot_attention(self):
        # 获取self.save_path目录下全部文件的名字
        import os
        npy_list = os.listdir(f'{self.save_path}/npy')
        npy_list_attn = [npy_str for npy_str in npy_list if 'attn' in npy_str]
        assert len(npy_list) == len(npy_list_attn)

        print(f'Start Plot: {self.save_path}, i: {len(npy_list_attn)}, j: 256')
        for idx in range(len(npy_list_attn)):
            # 加载数据
            attn_attn = np.load(f'{self.save_path}/npy/{idx}_attn.npy')
            # print(attn_attn.shape)          # (256, 336, 336)

            # 开始画图
            idx1 = attn_attn.shape[0] // 4
            idx2 = idx1 * 2
            idx3 = idx1 * 3
            os.makedirs(f'{self.save_path}/image', exist_ok=True)
            for jdx in [idx1, idx2, idx3]:
                attn_attn_ = attn_attn[jdx, :, :]

                # 归一化清洗
                attn_attn_ = self.my_clean(attn_attn_)

                import matplotlib.pyplot as plt
                plt.imshow(attn_attn_, aspect='auto')    # plt.colorbar()

                # 保存图片
                # plt.show()
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.png')
                plt.savefig(f'{self.save_path}/image/{idx}_{jdx}.svg')
                plt.close()

        return None

    def encode(self, x):
        # print('############# ShowModel.encode-1')
        B, N, D = x.shape

        # AttentionLayer
        # print(x.shape)                      # torch.Size([32, 126, 56])
        query = self.query_projection(x).view(B, N, self.n_head, -1)
        key = self.key_projection(x).view(B, N, self.n_head, -1)
        value = self.value_projection(x).view(B, N, self.n_head, -1)
        # print(query.shape)                  # torch.Size([32, 126, 8, 21])
        # print(key.shape)                    # torch.Size([32, 126, 8, 21])
        # print(value.shape)                  # torch.Size([32, 126, 8, 21])
        new_x, attn = self.attention(
            query,
            key,
            value,
        )
        # print(new_x.shape)                  # torch.Size([32, 126, 8, 21])
        new_x = new_x.view(B, N, -1)
        # print(new_x.shape)                  # torch.Size([32, 126, 168])
        new_x = self.out_projection(new_x)
        # print(new_x.shape)                  # torch.Size([32, 126, 56])
        x = x + self.dropout(new_x)
        # print(x.shape)                      # torch.Size([32, 126, 56])

        # print(attn.shape)                   # torch.Size([32, 336, 336])

        # FFNLayer
        x = self.norm1(x)
        new_x = x
        # print(new_x.shape)                  # torch.Size([32, 126, 56])
        new_x = self.dropout(self.activation(self.conv1(new_x.transpose(-1, 1))))
        # print(new_x.shape)                  # torch.Size([32, 126, 126])
        new_x = self.dropout(self.conv2(new_x).transpose(-1, 1))
        # print(new_x.shape)                  # torch.Size([32, 126, 56])
        x = self.norm2(x + new_x)
        # print(x.shape)                      # torch.Size([32, 126, 56])
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2]//self.patch_len+1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        # print(self.patch_len)       # 8
        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # print(c1.shape)             # torch.Size([32, 7, 42, 8])
        # print(c2.shape)             # torch.Size([32, 7, 42, 8])
        # print(c3.shape)             # torch.Size([32, 7, 42, 8])
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        # print(c1.shape)             # torch.Size([32, 42, 7, 8])
        # print(c2.shape)             # torch.Size([32, 42, 7, 8])
        # print(c3.shape)             # torch.Size([32, 42, 7, 8])
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        # print(c1.shape)             # torch.Size([32, 42, 56])
        # print(c2.shape)             # torch.Size([32, 42, 56])
        # print(c3.shape)             # torch.Size([32, 42, 56])
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([32, 126, 56])
        # print(attn.shape)           # torch.Size([32, 126, 126])

        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        # print(c.shape)              # torch.Size([32, 126, 7, 8])
        c = c.permute(0, 2, 1, 3).contiguous()
        # print(c.shape)              # torch.Size([32, 7, 126, 8])
        c = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        # print(c.shape)              # torch.Size([32, 7, 1008])
        o = self.projection(c)
        # print(o.shape)              # torch.Size([32, 7, 96])
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 96, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 96])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 96, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 96, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        c = c.permute(0, 2, 1, 3).contiguous()
        c = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        o = self.projection(c)
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 336, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 336, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 336, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        c = c.permute(0, 2, 1, 3).contiguous()
        c = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        o = self.projection(c)
        o = o.permute(0, 2, 1).contiguous()
        # print(o.shape)              # torch.Size([32, 336, 7])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])
        ot = ot.permute(0, 2, 1).contiguous()
        # print(ot.shape)             # torch.Size([32, 336, 7])

        o = ot + o
        o = self.normalize_layer(o, 'denorm')
        # print(o.shape)              # torch.Size([32, 336, 7])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
        ct = tmpq_dict['trend'][:, :, :x.shape[-1]]
        c1 = tmpq_dict['seasonal_1'][:, :, :x.shape[-1]]
        c2 = tmpq_dict['seasonal_2'][:, :, :x.shape[-1]]
        c3 = tmpq_dict['seasonal_3'][:, :, :x.shape[-1]]
        # print(ct.shape)             # torch.Size([32, 7, 336])
        # print(c1.shape)             # torch.Size([32, 7, 336])
        # print(c2.shape)             # torch.Size([32, 7, 336])
        # print(c3.shape)             # torch.Size([32, 7, 336])

        if x.shape[2] % self.patch_len != 0:
            padd_len = ((x.shape[2] // self.patch_len + 1) * self.patch_len) - x.shape[2]
            c1 = torch.cat([c1, torch.zeros((c1.shape[0], c1.shape[1], padd_len)).to(c1.device)], dim=2)
            c2 = torch.cat([c2, torch.zeros((c2.shape[0], c2.shape[1], padd_len)).to(c2.device)], dim=2)
            c3 = torch.cat([c3, torch.zeros((c3.shape[0], c3.shape[1], padd_len)).to(c3.device)], dim=2)

        c1 = c1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c2 = c2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c3 = c3.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        c1 = c1.permute(0, 2, 1, 3).contiguous()
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c3 = c3.permute(0, 2, 1, 3).contiguous()
        c1 = c1.reshape(c1.shape[0], c1.shape[1], -1).contiguous()
        c2 = c2.reshape(c2.shape[0], c2.shape[1], -1).contiguous()
        c3 = c3.reshape(c3.shape[0], c3.shape[1], -1).contiguous()
        c = torch.cat([c1, c2, c3], dim=1)
        # print(c.shape)              # torch.Size([32, 126, 56])

        c, attn = self.encode(c)
        # print(c.shape)              # torch.Size([32, 126, 56])
        # print(attn.shape)           # torch.Size([32, 126, 126])
        c = c.reshape(c.shape[0], c.shape[1], -1, self.patch_len).contiguous()
        # print(c.shape)              # torch.Size([32, 126, 7, 8])
        c = c.permute(0, 2, 1, 3).contiguous()
        # print(c.shape)              # torch.Size([32, 7, 126, 8])
        o = c.reshape(c.shape[0], c.shape[1], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 7, 1008])

        # Pattern:Trend
        # print(ct.shape)             # torch.Size([32, 7, 336])
        ot = self.up_ct(ct)
        # print(ot.shape)             # torch.Size([32, 7, 336])

        o = torch.cat([ot, o], 2)
        # print(o.shape)              # torch.Size([32, 7, 1344])

        # Output
        o = self.act(o)
        o = self.dropout(o)
        # print(o.shape)              # torch.Size([32, 7, 1344])
        o = o.reshape(o.shape[0], -1).contiguous()
        # print(o.shape)              # torch.Size([32, 9408])
        o = self.classifier(o)
        # print(o.shape)              # torch.Size([32, 4])

        if self.is_show:
            np.save(f'{self.save_path}/npy/{idx}_attn.npy', attn.cpu().detach().numpy())
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
