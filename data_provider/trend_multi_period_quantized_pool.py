"""
https://github.com/KishManani/MSTL/blob/main/mstl_decomposition.ipynb


conda source timer
pip install patsy==0.5.6
python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple statsmodels --upgrade --use-deprecated=legacy-resolver
pip show statsmodels

python test3.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# 输入: x(224,336,1)
# 输出: period_list(k,), period_weight(224,k)
# 计算全部224个序列的周期信息, 并返回能量最强的k个周期
def FFT_for_Period(x, k):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


# 输入: [166, 47, 23, 24, 49, 45]
# 输出: [[47, 49, 45], [166], [23, 24]]
def list_cluster(data, n_cluster):
    from sklearn.cluster import AgglomerativeClustering
    new_data = [[i, 1] for i in data]
    new_data = np.array(new_data)
    cluster_rst = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward').fit_predict(new_data)
    return_data = []
    for i in range(n_cluster):
        subData = new_data[cluster_rst == i]
        return_data.append(list(subData[:, 0]))
    return return_data


def plot_all_subfigure(dict):
    sub_num = len(dict)
    # plt.figure(figsize=(16, 4*(sub_num)), dpi=1000)

    for i, (key, value) in enumerate(dict.items()):
        plt.subplot(sub_num, 1, i+1)
        x = np.arange(1, len(value)+1)
        plt.plot(x, value)
        plt.ylabel(key)

    plt.show()
    return None


def TMPQ(y, period_num_choose_first=8, period_num_choose_last=3, period_num_max=3):
    import torch.nn as nn
    class moving_avg(nn.Module):
        def __init__(self, kernel_size, stride):
            super(moving_avg, self).__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        def forward(self, x):
            # print(self.kernel_size)
            # print(x.shape)                                          # torch.Size([224, 336, 1])     torch.Size([224, 336, 1])
            front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
            # print(front.shape)                                      # torch.Size([224, 55, 1])      torch.Size([224, 56, 1])
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            # print(end.shape)                                        # torch.Size([224, 55, 1])      torch.Size([224, 56, 1])
            x = torch.concat([front, x, end], dim=1)
            # print(x.shape)                                          # torch.Size([224, 446, 1])     torch.Size([224, 448, 1])
            x = self.avg(x.transpose(1, 2)).transpose(1, 2)
            # print(x.shape)                                          # torch.Size([224, 336, 1])     torch.Size([224, 336, 1])
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
                # print(x.shape)                          # torch.Size([224, 336, 1])     torch.Size([224, 336, 1])
                moving_avg = func(x)
                # print(moving_avg.shape)                 # torch.Size([224, 336, 1])     torch.Size([224, 336, 1])
                trend_.append(moving_avg.unsqueeze(0))
                sea = x - moving_avg
                # print(sea.shape)                        # torch.Size([224, 336, 1])     torch.Size([224, 336, 1])
                seasonal_.append(sea.unsqueeze(0))
            trend_ = torch.concat(trend_)
            seasonal_ = torch.concat(seasonal_)
            # print(seasonal_.shape)                      # torch.Size([2, 224, 336, 1])
            # print(trend_.shape)                         # torch.Size([2, 224, 336, 1])
            seasonal = torch.mean(seasonal_, axis=0)
            trend = torch.mean(trend_, axis=0)
            # print(seasonal.shape)                       # torch.Size([224, 336, 1])
            # print(trend.shape)                          # torch.Size([224, 336, 1])
            return seasonal, trend

    # 2. 第一层Trend-Seasonal分解, 将y分解为趋势部分trend1和季节部分y1
    # MSTL返回statsmodels.tsa.seasonal.DecomposeResult,
    # 其中包含5个属性: observed,_trend,_seasonal[0],_seasonal[1],_resid,
    # 这些属性满足关系: y == observed == (trend+seasonal1+seasonal2+resid)

    decomp = series_decomp_multi(kernel_size=(int(len(y)/2), int(len(y)/2)))
    y1, trend1 = decomp.forward(y)
    # print(y.shape)              # torch.Size([224, 336, 1])
    # print(y1.shape)             # torch.Size([224, 336, 1])

    # 3.1 根据FFT对季节部分y1分析多周期模式
    # 首先计算能量最强的前8个周期, 随后我们认为序列的周期不可能超过len(y1)/3, 随后筛掉过长的周期(大概率是残留的Trend部分)
    period_list, period_weight = FFT_for_Period(y1, k=period_num_choose_first)
    # print(period_list)          # [ 84 112  67 168  30  33   3   9]
    period_list_ = []
    for i in range(len(period_list)):
        if period_list[i] < (y1.shape[1] // period_num_max):
            period_list_.append(period_list[i])
    # print(period_list_)         # [67, 33, 8, 19, 8]

    # 3.2 将筛选后的一组潜在周期聚类为3组
    if period_list_:
        period_list__ = list_cluster(period_list_, period_num_choose_last)
    else:
        period_list__ = [[12], [24], [60]]
    # print(period_list__)        # [[8, 19, 8], [67], [33]]

    # 3.3 随后计算每组的中心, 并通过选取一组离散的周期长度以实现Period Quantized
    period_list___ = []
    for p_list in period_list__:
        p = np.mean(p_list)
        p = np.around(p / 4) if p >= 4 else 1
        p = int(p * 4)
        period_list___.append(p)
    period_list___.sort()
    # print(period_list___)       # [12, 32, 68]

    # 4. 第一层Trend-Seasonal分解, 将y1分解为趋势部分trend2和季节部分seasonal1, seasonal2, seasonal3
    decomp = series_decomp_multi(kernel_size=(int(period_list___[0]), int(period_list___[0])))
    y2, trend2 = decomp.forward(y1)
    decomp = series_decomp_multi(kernel_size=(int(period_list___[1]), int(period_list___[1])))
    y3, trend3 = decomp.forward(y2)
    decomp = series_decomp_multi(kernel_size=(int(period_list___[2]), int(period_list___[2])))
    y4, trend4 = decomp.forward(y3)

    plt_show = {}
    plt_show['observed'] = y
    plt_show['trend'] = trend1
    plt_show['seasonal1'] = trend2
    plt_show['seasonal2'] = trend3
    plt_show['seasonal3'] = trend4
    plt_show['resid'] = y4

    return plt_show


if __name__ == '__main__':
    # 0.1 Trend Multi-Period Quantized算法超参
    # 由于很难保证resid中不包含有效信息, 因此最终的period_num为period_num_choose_last+1, 因此我们取period_num_choose_last=2
    period_num_choose_first = 8     # 最先计算8个周期
    period_num_choose_last = 3      # 随后通过筛选、聚类后得到3个周期
    period_num_max = 3              # 对于长度为999的序列，我们认为其周期不可能超过999/3

    # 0.2 可视化超参
    pd.plotting.register_matplotlib_converters()
    plt.rc("figure", figsize=(8, 10))
    plt.rc("font", size=10)
    np.random.seed(0)

    # 1. 生成多周期时间序列
    t = np.arange(0, 1000)
    trend = 0.0001 * t ** 2 + 100
    daily_seasonality = 5 * np.sin(2 * np.pi * t / 24)
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / (24 * 7))
    weekly_seasonality_ = 10 * np.sin(2 * np.pi * t / (24 * 2))
    noise = np.random.randn(len(t))
    y = noise + trend + daily_seasonality + weekly_seasonality + weekly_seasonality_
    # print(y.shape)              # (999,)
    y = torch.from_numpy(y)
    # print(y.shape)              # torch.Size([999])

    # 5. 前向传播 & 可视化
    plt_show = TMPQ(y, period_num_choose_first, period_num_choose_last, period_num_max)
    plot_all_subfigure(plt_show)





