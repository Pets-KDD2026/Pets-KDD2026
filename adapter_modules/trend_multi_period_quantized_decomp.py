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
        time_series = dict['observed'][sample_idx, channel_idx, :]
        y_min, y_max = torch.min(time_series) - 0.5, torch.max(time_series) - 0.5
        value = value_[sample_idx, channel_idx, :]
        plt.subplot(sub_num, 1, i + 1)
        x = np.arange(1, len(value) + 1)
        plt.plot(x, value)
        plt.ylabel(key)
        if key != 'observed':
            plt.ylim((y_min, y_max))

    # plt.savefig(f'./fuck_{data_name}/sample_{sample_idx}_{channel_idx}.png')
    plt.show()
    # plt.close()
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


def fft_analysis(time_series, sampling_rate=1.0):
    """
    利用快速傅里叶变换（FFT）分析时间序列。

    参数:
    time_series (numpy.ndarray): 输入的时间序列。
    sampling_rate (float): 采样率，即每秒采样点数。

    返回:
    freqs (numpy.ndarray): 频率数组。
    amps (numpy.ndarray): 对应频率的振幅数组。
    """
    # 执行快速傅里叶变换
    fft_values = np.fft.fft(time_series)
    # 计算频率数组（只取正频率部分）
    n = len(time_series)
    freqs = np.fft.fftfreq(n, 1 / sampling_rate)
    freqs = freqs[:n // 2]  # 频率是对称的，只取一半
    # 计算振幅（取绝对值并乘以2/n，因为FFT结果是对称的，且包含实部和虚部）
    amps = 2.0 / n * np.abs(fft_values[:n // 2])

    return freqs, amps, fft_values


def TMPQ_single(time_series):
    freq_fft, fft_amps, fft_values = fft_analysis(time_series)

    # 步骤 5: 分解频谱
    idx_1 = max(enumerate(np.abs(fft_values[10:-10])), key=lambda x: x[1])[0]
    idx_1 = idx_1 + 10 if idx_1 < len(fft_values)//2 else idx_1 - 10
    idx_2 = len(fft_values) - 1 - idx_1
    idx_1_, idx_2_ = min(idx_1, idx_2), max(idx_1, idx_2)
    # print(f'{idx_1_}-{idx_2_}')
    fft_values_1 = np.array([fft_values[i] if (i <= idx_1_ or i >= idx_2_) else 0 for i in range(len(fft_values))])
    fft_values_23 = np.array([fft_values[i] if (i > idx_1_ and i < idx_2_) else 0 for i in range(len(fft_values))])

    idx_1 = max(enumerate(np.abs(fft_values_23[20:-20])), key=lambda x: x[1])[0]
    idx_1 = idx_1 + 20 if idx_1 < len(fft_values)//2 else idx_1 - 20
    idx_2 = len(fft_values_23) - 1 - idx_1
    idx_1_, idx_2_ = min(idx_1, idx_2), max(idx_1, idx_2)
    # print(f'{idx_1_}-{idx_2_}')
    fft_values_2 = np.array([fft_values_23[i] if (i <= idx_1_ or i >= idx_2_) else 0 for i in range(len(fft_values_23))])
    fft_values_3 = np.array([fft_values_23[i] if (i > idx_1_ and i < idx_2_) else 0 for i in range(len(fft_values_23))])

    # 步骤 6: 根据分解子频带重建时域组分
    fft_values_1_ = fft_values_1
    fft_values_2_ = fft_values_2
    fft_values_3_ = fft_values_3
    ifft_result_1 = np.fft.ifft(fft_values_1_)
    ifft_result_2 = np.fft.ifft(fft_values_2_)
    ifft_result_3 = np.fft.ifft(fft_values_3_)
    return ifft_result_1, ifft_result_2, ifft_result_3


def TMPQ(seq_x):
    # print(seq_x.shape)              # torch.Size([8209, 7, 336])

    # 步骤 1: 数据预处理
    decomp = series_decomp_multi(kernel_size=(int(seq_x.shape[2] / 2), int(seq_x.shape[2] / 2)))
    seq_x_seasonal, seq_x_trend = decomp.forward(seq_x)
    seq_x_seasonal = seq_x_seasonal.detach().cpu()
    # print(seq_x_trend.shape)        # torch.Size([8209, 7, 336])
    # print(seq_x_seasonal.shape)     # torch.Size([8209, 7, 336])

    """
    ifft_result_1 = torch.zeros_like(seq_x)
    ifft_result_2 = torch.zeros_like(seq_x)
    ifft_result_3 = torch.zeros_like(seq_x)

    for i in range(seq_x.shape[0]):
        for j in range(seq_x.shape[1]):
            s1, s2, s3 = TMPQ_single(seq_x_seasonal[i, j, :])
            ifft_result_1[i, j, :] = torch.tensor(np.real(s1)).to(seq_x.device)
            ifft_result_2[i, j, :] = torch.tensor(np.real(s2)).to(seq_x.device)
            ifft_result_3[i, j, :] = torch.tensor(np.real(s3)).to(seq_x.device)
    """

    # 步骤 2: 快速傅里叶变换
    _, _, fft_values = fft_analysis(seq_x_seasonal)
    # print(fft_values.shape)         # (8209, 7, 336)

    # 步骤 3.1: 分解频谱-1
    idx_1 = np.argmax(np.abs(fft_values[:, :, 10:-10]), axis=2)
    for i in range(idx_1.shape[0]):
        for j in range(idx_1.shape[1]):
            if idx_1[i, j] < fft_values.shape[2]//2:
                idx_1[i, j] += 10
            else:
                idx_1[i, j] -= 10
    idx_2 = fft_values.shape[2] - 1 - idx_1
    # print(idx_1.shape)              # (8209, 7)
    # print(idx_2.shape)              # (8209, 7)
    idx_1_, idx_2_ = np.minimum(idx_1, idx_2), np.maximum(idx_1, idx_2)
    # print(idx_1_.shape)             # (8209, 7)
    # print(idx_2_.shape)             # (8209, 7)

    fft_values_1 = np.zeros_like(fft_values)
    fft_values_23 = np.zeros_like(fft_values)
    for i in range(fft_values.shape[0]):
        for j in range(fft_values.shape[1]):
            for k in range(fft_values.shape[2]):
                if k <= idx_1_[i, j] or k >= idx_2_[i, j]:
                    fft_values_1[i, j, k] = fft_values[i, j, k]
                else:
                    fft_values_23[i, j, k] = fft_values[i, j, k]
    # print(fft_values_1.shape)       # (8209, 7, 336)
    # print(fft_values_23.shape)      # (8209, 7, 336)

    # 步骤 3.2: 分解频谱-2
    idx_1 = np.argmax(np.abs(fft_values_23[:, :, 20:-20]), axis=2)
    for i in range(idx_1.shape[0]):
        for j in range(idx_1.shape[1]):
            if idx_1[i, j] < fft_values.shape[2]//2:
                idx_1[i, j] += 20
            else:
                idx_1[i, j] -= 20
    idx_2 = fft_values.shape[2] - 1 - idx_1
    idx_1_, idx_2_ = np.minimum(idx_1, idx_2), np.maximum(idx_1, idx_2)
    fft_values_2 = np.zeros_like(fft_values)
    fft_values_3 = np.zeros_like(fft_values)
    for i in range(fft_values.shape[0]):
        for j in range(fft_values.shape[1]):
            for k in range(fft_values.shape[2]):
                if k <= idx_1_[i, j] or k >= idx_2_[i, j]:
                    fft_values_2[i, j, k] = fft_values_23[i, j, k]
                else:
                    fft_values_3[i, j, k] = fft_values_23[i, j, k]

    # 步骤 6: 根据分解子频带重建时域组分
    fft_values_1_ = fft_values_1
    fft_values_2_ = fft_values_2
    fft_values_3_ = fft_values_3
    ifft_result_1 = torch.tensor(np.fft.ifft(fft_values_1_)).to(seq_x.device)
    ifft_result_2 = torch.tensor(np.fft.ifft(fft_values_2_)).to(seq_x.device)
    ifft_result_3 = torch.tensor(np.fft.ifft(fft_values_3_)).to(seq_x.device)
    # print(ifft_result_1.shape)      # (8209, 7, 336)
    # print(ifft_result_2.shape)      # (8209, 7, 336)
    # print(ifft_result_3.shape)      # (8209, 7, 336)

    # 整理可视化结果
    plt_show = {}
    plt_show['observed'] = seq_x
    plt_show['trend'] = seq_x_trend
    plt_show['seasonal_1'] = ifft_result_1
    plt_show['seasonal_2'] = ifft_result_2
    plt_show['seasonal_3'] = ifft_result_3
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