"""
pip install PyWavelets
"""

import os
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def plot_all_subfigure(data_name, dict, sample_idx, channel_idx):
    sub_num = len(dict)

    for i, (key, value_) in enumerate(dict.items()):
        value = value_[sample_idx, channel_idx, :]
        plt.subplot(sub_num, 1, i+1)
        x = np.arange(1, len(value)+1)
        plt.plot(x, value)
        plt.ylabel(key)

    plt.savefig(f'./fuck_{data_name}/sample_{sample_idx}_{channel_idx}.png')
    # plt.show()
    plt.close()
    return None


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # print(x.shape)                      # torch.Size([8209, 7, 10])
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size // 2)
        # print(front.shape)                  # torch.Size([8209, 7, 10])
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        # print(end.shape)                    # torch.Size([8209, 7, 356])
        x = torch.concat([front, x, end], dim=2)
        # print(x.shape)                      # torch.Size([8209, 7, 336])
        x = self.avg(x)
        # print(x.shape)                      # torch.Size([8209, 7, 336])
        return x


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size=(13, 17)):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        trend_ = []
        seasonal_ = []
        for func in self.moving_avg:
            # print(x.shape)                  # torch.Size([8209, 7, 336])
            moving_avg = func(x)
            # print(moving_avg.shape)         # torch.Size([8209, 7, 336])
            trend_.append(moving_avg.unsqueeze(0))
            sea = x - moving_avg
            # print(sea.shape)                # torch.Size([8209, 7, 336])
            seasonal_.append(sea.unsqueeze(0))
        trend_ = torch.concat(trend_)
        seasonal_ = torch.concat(seasonal_)
        # print(seasonal_.shape)              # torch.Size([2, 8209, 7, 336])
        # print(trend_.shape)                 # torch.Size([2, 8209, 7, 336])
        seasonal = torch.mean(seasonal_, axis=0)
        trend = torch.mean(trend_, axis=0)
        # print(seasonal.shape)               # torch.Size([8209, 7, 336])
        # print(trend.shape)                  # torch.Size([8209, 7, 336])
        return seasonal, trend


def TMPQ(seq_x):
    if seq_x.shape[-1] <= 12:
        plt_show = {}
        plt_show['observed'] = seq_x
        plt_show['trend'] = seq_x
        plt_show['seasonal'] = seq_x
        plt_show['seasonal_1'] = seq_x
        plt_show['seasonal_2'] = seq_x
        plt_show['seasonal_3'] = seq_x
        return plt_show
    else:
        seq_len = seq_x.shape[-1]
        # print(seq_x.shape)                  # torch.Size([8209, 7, 336])

        # 趋势季节分解
        decomp = series_decomp_multi(kernel_size=(int(seq_x.shape[2] / 2), int(seq_x.shape[2] / 2)))
        seq_x_seasonal, seq_x_trend = decomp.forward(seq_x)
        # print(seq_x_trend.shape)            # torch.Size([8209, 7, 336])
        # print(seq_x_seasonal.shape)         # torch.Size([8209, 7, 336])

        """
        # 多周期分解
        # seq_x_list = pywt.wavedec(seq_x_seasonal, 'haar', level=2)
        # print(len(seq_x_list))              # 3
        # for seq_x_fre in seq_x_list:
        #     print(seq_x_fre.shape)          # (8209, 7, 84)     (8209, 7, 84)     (8209, 7, 168)
        """

        # 多周期分解
        c12, c3 = pywt.wavedec(seq_x_seasonal.detach().cpu(), 'haar', level=1)
        # print(c12.shape)                    # (8209, 7, 168)
        # print(c3.shape)                     # (8209, 7, 168)
        c12 = F.interpolate(torch.tensor(c12), scale_factor=2, mode='linear', align_corners=False,
                            recompute_scale_factor=True)
        c3 = F.interpolate(torch.tensor(c3), scale_factor=2, mode='linear', align_corners=False,
                           recompute_scale_factor=True).to(seq_x.device)
        # print(c12.shape)                    # torch.Size([8209, 7, 336])
        # print(c3.shape)                     # torch.Size([8209, 7, 336])
        c1, c2 = pywt.wavedec(c12, 'haar', level=1)
        # print(c1.shape)                     # (8209, 7, 168)
        # print(c2.shape)                     # (8209, 7, 168)
        c1 = F.interpolate(torch.tensor(c1), scale_factor=2, mode='linear', align_corners=False,
                           recompute_scale_factor=True).to(seq_x.device)
        c2 = F.interpolate(torch.tensor(c2), scale_factor=2, mode='linear', align_corners=False,
                           recompute_scale_factor=True).to(seq_x.device)
        # print(c1.shape)                     # torch.Size([8209, 7, 336])
        # print(c2.shape)                     # torch.Size([8209, 7, 336])
        seq_x_list = [c1, c2, c3]

        # 逆多周期分解
        # seq_x_time = pywt.waverec(seq_x_list, 'haar')
        # print(seq_x_time.shape)             # (8209, 7, 336)

        # 整理可视化结果
        plt_show = {}
        plt_show['observed'] = seq_x[:, :, :seq_len]
        plt_show['trend'] = seq_x_trend[:, :, :seq_len]
        plt_show['seasonal'] = seq_x_seasonal[:, :, :seq_len]
        for i in range(len(seq_x_list)):
            plt_show[f'seasonal_{i + 1}'] = seq_x_list[i][:, :, :seq_len]
        return plt_show


if __name__ == '__main__':
    data_name_list = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'traffic', 'electricity']
    data_name = 'ETTh1'

    # 设置画布
    plt.rc("figure", figsize=(8, 8))
    plt.rc("font", size=10)

    # 加载真实数据
    dir = f'./data_tmpq/_{data_name}_336/336_0_96_train'
    os.makedirs(f'./fuck_{data_name}', exist_ok=True)
    seq_x = np.load(f'{dir}/seq_x.npy')
    # print(seq_x.shape)                  # (8209, 336, 7)

    # 转换为tensor，并且确保形状为(batch_size,channels,seq_len)
    seq_x = torch.tensor(seq_x).permute(0, 2, 1).contiguous()
    # print(seq_x.shape)                  # torch.Size([8209, 7, 336])

    plt_show = TMPQ(seq_x)

    # 画图
    for sample_idx_ in range(int(seq_x.shape[0] / seq_x.shape[2])):
        for channel_idx in range(seq_x.shape[1]):
            sample_idx = sample_idx_ * seq_x.shape[2]
            print(f'{data_name}_{sample_idx}_{channel_idx}')

            plot_all_subfigure(data_name, plt_show, sample_idx, channel_idx)