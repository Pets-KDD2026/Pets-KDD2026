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

from statsmodels.tsa.seasonal import MSTL

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

    plt.savefig('./tmpq_sample.png')
    plt.show()
    return None


def TMPQ_single(y, period_num_choose_first=8, period_num_choose_last=3, period_num_max=3):

    # 2. 第一层Trend-Seasonal分解, 将y分解为趋势部分trend1和季节部分y1
    # MSTL返回statsmodels.tsa.seasonal.DecomposeResult,
    # 其中包含5个属性: observed,_trend,_seasonal[0],_seasonal[1],_resid,
    # 这些属性满足关系: y == observed == (trend+seasonal1+seasonal2+resid)
    # print(y.shape)              # torch.Size([336])

    mstl1 = MSTL(y, periods=(len(y) // 4, len(y) // 2)).fit()
    y1 = y - mstl1._trend
    # print(y.shape)              # torch.Size([336])
    # print(y1.shape)             # torch.Size([336])

    # 3.1 根据FFT对季节部分y1分析多周期模式
    # 首先计算能量最强的前8个周期, 随后我们认为序列的周期不可能超过len(y1)/3, 随后筛掉过长的周期(大概率是残留的Trend部分)
    period_list, period_weight = FFT_for_Period(y1.unsqueeze(0).unsqueeze(2), k=period_num_choose_first)
    # print(period_list)          # [999 166  47  23 499  24  49  45]
    period_list_ = []
    for i in range(len(period_list)):
        if period_list[i] < (len(y1) // period_num_max):
            period_list_.append(period_list[i])
    # print(period_list_)         # [166, 47, 23, 24, 49, 45]

    # 3.2 将筛选后的一组潜在周期聚类为3组
    period_list__ = list_cluster(period_list_, period_num_choose_last)
    # print(period_list__)        # [[47, 49, 45], [166], [23, 24]]

    # 3.3 随后计算每组的中心, 并通过选取一组离散的周期长度以实现Period Quantized
    period_list___ = []
    for p_list in period_list__:
        p = np.mean(p_list)
        p = np.around(p / 4) if p >= 4 else 1
        p = int(p * 4)
        period_list___.append(p)
    period_list___.sort()
    # print(period_list___)       # [24, 48, 168]

    # 4. 第一层Trend-Seasonal分解, 将y1分解为趋势部分trend2和季节部分seasonal1, seasonal2, seasonal3
    mstl2 = MSTL(y1, periods=period_list___).fit()
    # mstl2.plot()
    # plt.tight_layout()
    # plt.show()

    plt_show = {}
    plt_show['observed'] = y
    plt_show['trend'] = mstl1._trend + mstl2._trend
    for i in range(mstl2._seasonal.shape[1]):
        plt_show[f'seasonal{i + 1}'] = mstl2._seasonal[:, i]
    plt_show['resid'] = mstl2._resid
    # return plt_show
    return y, mstl1._trend+mstl2._trend, mstl2._seasonal[:, 0], mstl2._seasonal[:, 1], mstl2._seasonal[:, 2], mstl2._resid


def TMPQ(y, period_num_choose_first=8, period_num_choose_last=3, period_num_max=3):

    # 2. 第一层Trend-Seasonal分解, 将y分解为趋势部分trend1和季节部分y1
    # MSTL返回statsmodels.tsa.seasonal.DecomposeResult,
    # 其中包含5个属性: observed,_trend,_seasonal[0],_seasonal[1],_resid,
    # 这些属性满足关系: y == observed == (trend+seasonal1+seasonal2+resid)
    # print(y.shape)              # torch.Size([224, 336, 1])
    trend = np.zeros(y.shape)
    seasonal1 = np.zeros(y.shape)
    seasonal2 = np.zeros(y.shape)
    seasonal3 = np.zeros(y.shape)
    resid = np.zeros(y.shape)

    for i in range(y.shape[0]):
        for j in range(y.shape[2]):
            y_ = y[i, :, j]
            observed_, trend_, seasonal1_, seasonal2_, seasonal3_, resid_ = TMPQ_single(
                y_, period_num_choose_first, period_num_choose_last, period_num_max
            )
            trend[i, :, j] = trend_
            seasonal1[i, :, j] = seasonal1_
            seasonal2[i, :, j] = seasonal2_
            seasonal3[i, :, j] = seasonal3_
            resid[i, :, j] = resid_

    trend = torch.from_numpy(trend).float().cuda().squeeze(2)
    seasonal1 = torch.from_numpy(seasonal1).float().cuda().squeeze(2)
    seasonal2 = torch.from_numpy(seasonal2).float().cuda().squeeze(2)
    seasonal3 = torch.from_numpy(seasonal3).float().cuda().squeeze(2)
    resid = torch.from_numpy(resid).float().cuda().squeeze(2)
    # print(trend.shape)          # torch.Size([224, 336, 1])
    # print(seasonal1.shape)      # torch.Size([224, 336, 1])
    # print(seasonal2.shape)      # torch.Size([224, 336, 1])
    # print(seasonal3.shape)      # torch.Size([224, 336, 1])
    # print(resid.shape)          # torch.Size([224, 336, 1])
    return trend, seasonal1, seasonal2, seasonal3, resid


if __name__ == '__main__':
    # 0.1 Trend Multi-Period Quantized算法超参
    # 由于很难保证resid中不包含有效信息, 因此最终的period_num为period_num_choose_last+1, 因此我们取period_num_choose_last=2
    period_num_choose_first = 8  # 最先计算8个周期
    period_num_choose_last = 3  # 随后通过筛选、聚类后得到3个周期
    period_num_max = 3  # 对于长度为999的序列，我们认为其周期不可能超过999/3

    # 0.2 可视化超参
    pd.plotting.register_matplotlib_converters()
    plt.rc("figure", figsize=(8, 10))
    plt.rc("font", size=10)
    np.random.seed(0)

    # 1. 生成多周期时间序列
    t = np.arange(1000)
    trend = 0.0001 * t ** 2 + 100
    daily_seasonality = 5 * np.sin(2 * np.pi * t / 24)
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / (24 * 6))
    weekly_seasonality_ = 10 * np.sin(2 * np.pi * t / (24 * 2))
    noise = np.random.randn(len(t))
    y = noise + trend + daily_seasonality + weekly_seasonality + weekly_seasonality_
    # print(y.shape)              # (999,)
    y = torch.from_numpy(y)
    # print(y.shape)              # torch.Size([999])

    # 5. 前向传播 & 可视化
    plt_show = TMPQ_single(y, period_num_choose_first, period_num_choose_last, period_num_max)
    plot_all_subfigure(plt_show)





