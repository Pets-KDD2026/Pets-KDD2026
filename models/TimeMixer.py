import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print("#################################### TokenEmbedding-1 #####################################")
        #                                                       TimesNet                    PatchTST
        # print(x.shape)                                          # torch.Size([32, 96, 7])   torch.Size([224, 12, 16])
        # print(x.permute(0, 2, 1).shape)                         # torch.Size([32, 7, 96])   torch.Size([224, 16, 12])
        # print(self.tokenConv(x.permute(0, 2, 1)).shape)         # torch.Size([32, 16, 96])  torch.Size([224, 16, 12])
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # print(x.shape)                                          # torch.Size([32, 96, 16])  torch.Size([224, 12, 16])
        # print("#################################### TokenEmbedding-2 #####################################")
        return x


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = \
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else \
            TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # print('############# DataEmbedding_wo_pos-1')
        # print(x.shape)                          # torch.Size([896, 336, 1])
        # print(x_mark)                           # None
        if x_mark is None:
            # print(type(self.value_embedding))   # <class 'models.TimeMixer.TokenEmbedding'>
            # print(x.shape)                      # torch.Size([896, 336, 1])
            x = self.value_embedding(x)
            # print(x.shape)                      # torch.Size([896, 336, 16])
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        # print('############# DataEmbedding_wo_pos-2')
        return self.dropout(x)


# Moving average block to highlight the trend of time series
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # print('############# moving_avg-1')
        # print(self.kernel_size)     # 25
        # print(x.shape)              # torch.Size([896, 336, 16])
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # print(front.shape)          # torch.Size([896, 12, 16])
        # print(end.shape)            # torch.Size([896, 12, 16])
        x = torch.cat([front, x, end], dim=1)
        # print(x.shape)              # torch.Size([896, 360, 16])
        x = x.permute(0, 2, 1)
        # print(x.shape)              # torch.Size([896, 16, 360])
        x = self.avg(x)
        # print(x.shape)              # torch.Size([896, 16, 336])
        x = x.permute(0, 2, 1)
        # print(x.shape)              # torch.Size([896, 336, 16])
        # print('############# moving_avg-2')
        return x


# Series decomposition block
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # print('############# series_decomp-1')
        # print(type(self.moving_avg))        # <class 'models.TimeMixer.moving_avg'>
        # print(x.shape)                      # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
        moving_mean = self.moving_avg(x)
        # print(moving_mean.shape)            # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
        res = x - moving_mean
        # print(res.shape)                    # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
        # print('############# series_decomp-2')
        return res, moving_mean


# Discrete Fourier decomposition block
class DFT_series_decomp(nn.Module):
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        print('############# DFT_series_decomp-1')
        print(x.shape)
        xf = torch.fft.rfft(x)
        print(xf.shape)
        freq = abs(xf)
        print(freq.shape)

        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        print(top_k_freq)
        print(top_list)

        xf[freq <= top_k_freq.min()] = 0
        print(xf.shape)
        x_season = torch.fft.irfft(xf)
        print(x_season.shape)
        x_trend = x - x_season
        print(x_trend)
        print('############# DFT_series_decomp-2')
        return x_season, x_trend


# Bottom-up (high->low) mixing season pattern
# 输入一组多尺度季节序列: (896,16,336), (896,16,168), (896,16,84), (896,16,42)
# 对于两个相邻尺度的序列out_high(336)和out_low(168), 基于MLP_336_168将out_high下采样到长度为168, 并与out_low相加作为新的out_low，
# 随后为接下来的相邻尺度序列重复这一操作, 从而将(896,16,168),(896,16,84),(896,16,42)这三个尺度的序列都融入了更大尺度的【季节时序信息】,
# 最后将原始的(896,16,336)和更新后的(896,16,168),(896,16,84),(896,16,42)组成列表并返回
class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = torch.nn.ModuleList([
            nn.Sequential(
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
            ) for i in range(configs.down_sampling_layers)
        ])

    # 输入一组多尺度季节序列: (896,16,336), (896,16,168), (896,16,84), (896,16,42)
    # 对于两个相邻尺度的序列out_high(336)和out_low(168), 基于MLP_336_168将out_high下采样到长度为168, 并与out_low相加作为新的out_low，
    # 随后为接下来的相邻尺度序列重复这一操作, 从而将(896,16,168),(896,16,84),(896,16,42)这三个尺度的序列都融入了更大尺度的时序信息,
    # 最后将原始的(896,16,336)和更新后的(896,16,168),(896,16,84),(896,16,42)组成列表并返回
    def forward(self, season_list):
        # print('############# MultiScaleSeasonMixing-1')
        # print(len(season_list))     # 4
        # for tensor_ in season_list:
        #     print(tensor_.shape)    # torch.Size([896, 16, 336])    torch.Size([896, 16, 168])  torch.Size([896, 16, 84])   torch.Size([896, 16, 42])

        out_high = season_list[0]
        out_low = season_list[1]
        # print(out_high.shape)       # torch.Size([896, 16, 336])
        # print(out_low.shape)        # torch.Size([896, 16, 168])

        # print(len(season_list))     # 4
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            # print(i)                # 0                             1                           2
            # print(type(self.down_sampling_layers[i]))   # <class 'torch.nn.modules.container.Sequential'>
            # print(out_high.shape)   # torch.Size([896, 16, 336])    torch.Size([896, 16, 168])  torch.Size([896, 16, 84])
            out_low_res = self.down_sampling_layers[i](out_high)
            # print(out_low_res.shape)# torch.Size([896, 16, 168])    torch.Size([896, 16, 84])   torch.Size([896, 16, 42])
            out_low = out_low + out_low_res
            # print(out_low.shape)    # torch.Size([896, 16, 168])    torch.Size([896, 16, 84])   torch.Size([896, 16, 42])
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                # print('fuck')       # fuck                          fuck
                out_low = season_list[i + 2]
                # print(out_low.shape)# torch.Size([896, 16, 84])     torch.Size([896, 16, 42])
            # print(out_high.shape)   # torch.Size([896, 16, 168])    torch.Size([896, 16, 84])   torch.Size([896, 16, 42])
            out_season_list.append(out_high.permute(0, 2, 1))

        # print('############# MultiScaleSeasonMixing-2')
        return out_season_list


# Top->down (low->high) mixing trend pattern
# 输入一组多尺度季节序列: (896,16,42), (896,16,84), (896,16,168), (896,16,336)
# 对于两个相邻尺度的序列out_low(42)和out_high(84), 基于MLP_42_84将out_low上采样到长度为84, 并与out_high相加作为新的out_high，
# 随后为接下来的相邻尺度序列重复这一操作, 从而将(896,16,84),(896,16,168),(896,16,336)这三个尺度的序列都融入了更小尺度的【趋势时序信息】,
# 最后将原始的(896,16,42)和更新后的(896,16,84),(896,16,168),(896,16,336)组成列表并返回
class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList([
            nn.Sequential(
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
                nn.GELU(),
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
            ) for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list):
        # print('############# MultiScaleTrendMixing-1')

        # print(len(trend_list))          # 4
        # for tensor_ in trend_list:
        #     print(tensor_.shape)        # torch.Size([896, 16, 336])    torch.Size([896, 16, 168])  torch.Size([896, 16, 84])   torch.Size([896, 16, 42])
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        # print(len(trend_list_reverse))  # 4
        # for tensor_ in trend_list_reverse:
        #     print(tensor_.shape)        # torch.Size([896, 16, 42])     torch.Size([896, 16, 84])   torch.Size([896, 16, 168])  torch.Size([896, 16, 336])

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        # print(out_low.shape)            # torch.Size([896, 16, 42])
        # print(out_high.shape)           # torch.Size([896, 16, 84])

        # print(len(trend_list_reverse))  # 4
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            # print(i)                    # 0                             1                           2
            # print(type(self.up_sampling_layers[i])) # <class 'torch.nn.modules.container.Sequential'>
            # print(out_low.shape)        # torch.Size([896, 16, 42])     torch.Size([896, 16, 84])   torch.Size([896, 16, 168])
            out_high_res = self.up_sampling_layers[i](out_low)
            # print(out_high_res.shape)   # torch.Size([896, 16, 84])     torch.Size([896, 16, 168])  torch.Size([896, 16, 336])
            # print(out_high.shape)       # torch.Size([896, 16, 84])     torch.Size([896, 16, 168])  torch.Size([896, 16, 336])
            out_high = out_high + out_high_res
            # print(out_high.shape)       # torch.Size([896, 16, 84])     torch.Size([896, 16, 168])  torch.Size([896, 16, 336])
            out_low = out_high
            # print(out_low.shape)        # torch.Size([896, 16, 84])     torch.Size([896, 16, 168])  torch.Size([896, 16, 336])
            if i + 2 <= len(trend_list_reverse) - 1:
                # print('fuck')           # fuck                          fuck
                out_high = trend_list_reverse[i + 2]
                # print(out_high.shape)   # torch.Size([896, 16, 168])    torch.Size([896, 16, 336])
            # print(out_low.shape)        # torch.Size([896, 16, 84])     torch.Size([896, 16, 168])  torch.Size([896, 16, 336])
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        # print('############# MultiScaleTrendMixing-2')
        return out_trend_list

# Adapter Embedding:
#  x1(896,336,1),  x2(896,168,1),  x3(896,84,1),  x4(896,42,1) -TMPQ->
# x11(896,336,1), x21(896,168,1), x31(896,84,1), x41(896,42,1) +
# x12(896,336,1), x22(896,168,1), x32(896,84,1), x42(896,42,1) +
# x13(896,336,1), x23(896,168,1), x33(896,84,1), x43(896,42,1) -Embedding->
# e11(896,336,16),e21(896,168,16),e31(896,84,16),e41(896,42,16) +
# e12(896,336,16),e22(896,168,16),e32(896,84,16),e42(896,42,16) +
# e13(896,336,16),e23(896,168,16),e33(896,84,16),e43(896,42,16) -Concat->
#  c1(896,336,48), c2(896,168,48), c3(896,84,48), c4(896,42,48)

# Adapter Encoder
# CrossLayer_down_1: c1(896,336,48) -> c1(896,336,16)
# CrossLayer_down_2: c2(896,168,48) -> c2(896,168,16)
# CrossLayer_down_3: c3(896,84,48)  -> c3(896,84,16)
# CrossLayer_down_4: c4(896,42,48)  -> c4(896,42,16)
# CrossLayer_up_1: c1(896,336,16) -> c1(896,336,48)
# CrossLayer_up_2: c2(896,168,16) -> c2(896,168,48)
# CrossLayer_up_3: c3(896,84,16)  -> c3(896,84,48)
# CrossLayer_up_4: c4(896,42,16)  -> c4(896,42,48)


# PastDecomposableMixing as Encoder
# 输入一层多尺度原始序列, x1(896,336,16), x2(896,168,16), x3(896,84,16), x4(896,42,16)
#      季节趋势分解得到, s1(896,336,16), s2(896,168,16), s3(896,84,16), s4(896,42,16)
#      季节趋势分解得到, t1(896,336,16), t2(896,168,16), t3(896,84,16), t4(896,42,16)
#      在(s1,s2,s3,s4)之间融合信息建立连接, 在(t1,t2,t3,t4)之间融合信息建立连接,
#      残差连接计算, o1=x1+MLP(s1+t1), o2=x2+MLP(s2+t2), o3=x3+MLP(s3+t3), o4=x4+MLP(s4+t4),
# 输出一层多尺度原始序列, o1(896,336,16), o2(896,168,16), o3(896,84,16), o4(896,42,16)
class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        # print('############# PastDecomposableMixing-1')
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # 1. 首先将4条【多尺度原始序列】分别经过季节-趋势分解得到4条【多尺度季节序列】和4条【多尺度趋势序列】
        # print(len(x_list))                      # 4
        season_list = []
        trend_list = []
        for x in x_list:
            # print(type(self.decompsition))      # <class 'models.TimeMixer.series_decomp'>
            # print(x.shape)                      # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
            season, trend = self.decompsition(x)
            # print(season.shape)                 # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
            # print(trend.shape)                  # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])

            # print(self.channel_independence)    # 1
            if not self.channel_independence:
                # print(type(self.cross_layer))   # <class 'torch.nn.modules.container.Sequential'>
                # print(season.shape)             # torch.Size([32, 336, 16])     torch.Size([32, 168, 16])   torch.Size([32, 84, 16])    torch.Size([32, 42, 16])
                # print(trend.shape)              # torch.Size([32, 336, 16])     torch.Size([32, 168, 16])   torch.Size([32, 84, 16])    torch.Size([32, 42, 16])
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
                # print(season.shape)             # torch.Size([32, 336, 16])     torch.Size([32, 168, 16])   torch.Size([32, 84, 16])    torch.Size([32, 42, 16])
                # print(trend.shape)              # torch.Size([32, 336, 16])     torch.Size([32, 168, 16])   torch.Size([32, 84, 16])    torch.Size([32, 42, 16])

            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # 2. MultiScaleSeasonMixing从4条多尺度季节序列中提取深层特征
        # bottom-up season mixing
        # print(type(self.mixing_multi_scale_season))# <class 'models.TimeMixer.MultiScaleSeasonMixing'>
        # print(len(season_list))                 # 4
        # for tensor_ in season_list:
        #     print(tensor_.shape)                # torch.Size([896, 16, 336])    torch.Size([896, 16, 168])  torch.Size([896, 16, 84])   torch.Size([896, 16, 42])
        out_season_list = self.mixing_multi_scale_season(season_list)
        # print(len(out_season_list))             # 4
        # for tensor_ in out_season_list:
        #     print(tensor_.shape)                # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])

        # 3. MultiScaleTrendMixing从4条多尺度趋势序列中提取深层特征
        # top-down trend mixing
        # print(type(self.mixing_multi_scale_trend))# <class 'models.TimeMixer.MultiScaleTrendMixing'>
        # print(len(trend_list))                  # 4
        # for tensor_ in trend_list:
        #     print(tensor_.shape)                # torch.Size([896, 16, 336])    torch.Size([896, 16, 168])  torch.Size([896, 16, 84])   torch.Size([896, 16, 42])
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        # print(len(out_trend_list))              # 4
        # for tensor_ in out_trend_list:
        #     print(tensor_.shape)                # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])

        # 4. 对于每个尺度, 获取原始序列ori, 季节序列out_season, 趋势序列out_trend
        # 使用out=ori+MLP(out_season+out_trend)作为原始序列ori的深层表征
        # print(self.channel_independence)        # 1
        # print(len(x_list))                      # 4
        # print(len(out_season_list))             # 4
        # print(len(out_trend_list))              # 4
        # print(len(length_list))                 # 4
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            # print(out_season.shape)             # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
            # print(out_trend.shape)              # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
            out = out_season + out_trend
            # print(out.shape)                    # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
            if self.channel_independence:
                # print(type(self.out_cross_layer))# <class 'torch.nn.modules.container.Sequential'>
                # print(out.shape)                # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
                # print(ori.shape)                # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
                out = ori + self.out_cross_layer(out)
                # print(out.shape)                # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
            out_list.append(out[:, :length, :])

        # print('############# PastDecomposableMixing-2')
        return out_list


# self.normalize_layer = Normalize(configs.enc_in, affine=True, non_norm=False)
# print(x.shape)          # torch.Size([128, 336, 7])
# x = self.normalize_layer(x, 'norm')
# print(x.shape)          # torch.Size([128, 336, 7])
# print(x.shape)          # torch.Size([128, 96, 7])
# x = self.normalize_layer(x, 'denorm')
# print(x.shape)          # torch.Size([128, 96, 7])
class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# Encoder
# 1. 首先将4条【多尺度原始序列ori_list】分别经过季节-趋势分解得到4条【多尺度季节序列season_list】和4条【多尺度趋势序列trend_list】
#    首先将长度为336的原始序列拆解多尺度分量336,168,84,42, 有点类似于多层的离散小波变换
#    输入原始序列(128,336,7), 经过连续的3层AvgPool1d得到3个不同尺度的一组序列组(128,168,7), (128,84,7), (128,42,7)
#    返回由全部4个序列组成的列表x1(128,336,7), x2(128,168,7), x3(128,84,7), x4(128,42,7)
#
# 2. 基于RevIn的Normalization预处理 & Channel-Independency
#    全部4个尺度, 每个尺度都需要一个独立的normalize_layer做预处理, 最后仅使用原始输入序列对应的那个normalize_layer做DeNorm
#
# 3. 不同尺度的子序列{x1,x2,x3,x4}分别经过相同的Temporal Embedding, 注意这里采用的是类似于TimesNet的嵌入方式, 而非PatchTST的嵌入方式
#    多尺度嵌入为, e1(896,336,16), e2(896,168,16), e3(896,84,16), e4(896,42,16)
#
# 4. PastDecomposableMixing as Encoder
# 4.1 MultiScaleSeasonMixing从4条【多尺度季节序列season_list】中提取4条【多尺度季节深层特征season_out_list】
# 4.2 MultiScaleTrendMixing从4条【多尺度趋势序列trend_list】中提取4条【多尺度趋势深层特征trend_out_list】
# 4.3 对于每个尺度, 获取原始序列ori, 季节序列season_out, 趋势序列trend_out,
#     使用out=ori+MLP(out_season+out_trend)作为原始序列ori的深层表征, 对于全部4个尺度得到out_list
#     多尺度深层特征为, e1(896,336,16), e2(896,168,16), e3(896,84,16), e4(896,42,16)
#
# 5. PastDecomposableMixing as Decoder
#    经过两层MLP映射层, channel_projection,sequence_projection, 将e1,e2,e3,e4下采样到(896,96,1), 最后将4个(96,)求和作为预测结果
#
# 6. DeNormalization & De-Channel-Independency
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # print('############# TimeMixer-1')
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        # self.channel_independence = configs.channel_independence
        self.channel_independence = 0 if self.task_name == 'classification' else 1
        self.layer = configs.e_layers
        # print(self.task_name)               # long_term_forecast
        # print(self.seq_len)                 # 336
        # print(self.pred_len)                # 96
        # print(self.down_sampling_window)    # 2
        # print(self.channel_independence)    # 1

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs)for _ in range(configs.e_layers)
        ])
        # print(self.layer)                   # 2
        # print(type(PastDecomposableMixing)) # <class 'type'>

        self.preprocess = series_decomp(configs.moving_avg)
        # print(type(self.preprocess))        # <class 'models.TimeMixer.series_decomp'>

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # print(self.channel_independence)    # 1
        # print(configs.enc_in)               # 7
        # print(configs.d_model)              # 16
        # print(configs.embed)                # timeF
        # print(configs.freq)                 # h
        # print(configs.dropout)              # 0.1

        self.normalize_layers = torch.nn.ModuleList([
            Normalize(
                configs.enc_in,
                affine=True,
                non_norm=True if configs.use_norm == 0 else False
            ) for _ in range(configs.down_sampling_layers + 1)
        ])
        # print(configs.down_sampling_layers) # 3
        # print(configs.enc_in)               # 7
        # print(configs.use_norm)             # 1

        # Multi-Linear
        # predict_layers    :  seq_len//(down_window**i) -> pred_len
        # projection_layer  :  d_model -> c_out
        # out_res_layers    :  seq_len//(down_window**i) -> seq_len//(down_window**i)
        # regression_layers :  seq_len//(down_window**i) -> pred_len
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                ) for i in range(configs.down_sampling_layers + 1)
            ])

            if self.channel_independence:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ) for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    ) for i in range(configs.down_sampling_layers + 1)
                ])

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        # print('############# TimeMixer-2')

    def pre_enc(self, x_list):
        # print('############# TimeMixer.pre_enc-1')
        # print(self.channel_independence)    # 1
        if self.channel_independence:
            # print('############# TimeMixer.pre_enc-2')
            return x_list, None
        else:
            # print('############# TimeMixer.pre_enc-1')
            # print(len(x_list))      # 4
            out1_list = []
            out2_list = []
            for x in x_list:
                # print(type(self.preprocess))    # <class 'models.TimeMixer.series_decomp'>
                # print(x.shape)      # torch.Size([32, 336, 7])   torch.Size([32, 168, 7])   torch.Size([32, 84, 7])    torch.Size([32, 42, 7])
                x_1, x_2 = self.preprocess(x)
                # print(x_1.shape)    # torch.Size([32, 336, 7])   torch.Size([32, 168, 7])   torch.Size([32, 84, 7])    torch.Size([32, 42, 7])
                # print(x_2.shape)    # torch.Size([32, 336, 7])   torch.Size([32, 168, 7])   torch.Size([32, 84, 7])    torch.Size([32, 42, 7])
                out1_list.append(x_1)
                out2_list.append(x_2)
            # print('############# TimeMixer.pre_enc-3')
            return out1_list, out2_list

    def __multi_scale_process_inputs(self, x_enc):
        # print('############# TimeMixer.multi_scale_process_inputs-1')
        # print(self.configs.down_sampling_method)# avg

        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=(3,),
                padding=padding,
                stride=self.configs.down_sampling_window,
                padding_mode='circular',
                bias=False
            )
        else:
            return x_enc
        # print(type(down_pool))                  # <class 'torch.nn.modules.pooling.AvgPool1d'>

        # print(x_enc.shape)                      # torch.Size([128, 336, 7])
        x_enc = x_enc.permute(0, 2, 1)
        # print(x_enc.shape)                      # torch.Size([128, 7, 336])
        x_enc_ori = x_enc

        # print(self.configs.down_sampling_layers)# 3
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.configs.down_sampling_layers):
            # print(type(down_pool))              # <class 'torch.nn.modules.pooling.AvgPool1d'>
            # print(x_enc_ori.shape)              # (128,7,336)    (128,7,168)    (128,7,84)
            x_enc_sampling = down_pool(x_enc_ori)
            # print(x_enc_sampling.shape)         # (128,7,168)    (128,7,84)     (128,7,42)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        x_enc = x_enc_sampling_list
        # print(len(x_enc))                       # 4
        # for tensor_ in x_enc:
        #     print(tensor_.shape)                # (128,336,7)   (128,168,7)    (128,84,7)     (128,42,7)
        # print('############# TimeMixer.multi_scale_process_inputs-2')
        return x_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print('############# TimeMixer.forecast-1')

        # 1. 首先将长度为336的原始序列拆解多尺度分量336,168,84,42, 有点类似于多层的离散小波变换
        # 输入原始序列(128,336,7), 经过连续的3层AvgPool1d得到3个不同尺度的一组序列组(128,168,7), (128,84,7), (128,42,7)
        # 返回由全部4个序列组成的列表x1(128,336,7), x2(128,168,7), x3(128,84,7), x4(128,42,7)
        # print(x_enc.shape)              # torch.Size([128, 336, 7])
        x_enc_list = self.__multi_scale_process_inputs(x_enc)
        # print(len(x_enc_list))          # 4
        # for tensor_ in x_enc_list:
        #     print(tensor_.shape)        # torch.Size([128, 336, 7]) torch.Size([128, 168, 7])   torch.Size([128, 84, 7])    torch.Size([128, 42, 7])

        # 2. 基于RevIn的Normalization预处理 & Channel-Independency
        #    全部4个尺度, 每个尺度都需要一个独立的normalize_layer做预处理, 最后仅使用原始输入序列对应的那个normalize_layer做DeNorm
        x_list = []
        # print(self.channel_independence)# 1
        for i, x in enumerate(x_enc_list):
            # print(i)                    # 0                         1                           2                           3
            B, T, N = x.size()
            # print(type(self.normalize_layers[i]))   # <class 'models.TimeMixer.Normalize'>
            # print(x.shape)              # torch.Size([128, 336, 7]) torch.Size([128, 168, 7])   torch.Size([128, 84, 7])    torch.Size([128, 42, 7])
            x = self.normalize_layers[i](x, 'norm')
            # print(x.shape)              # torch.Size([128, 336, 7]) torch.Size([128, 168, 7])   torch.Size([128, 84, 7])    torch.Size([128, 42, 7])
            if self.channel_independence:
                # print(x.shape)          # torch.Size([128, 336, 7]) torch.Size([128, 168, 7])   torch.Size([128, 84, 7])    torch.Size([128, 42, 7])
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                # print(x.shape)          # torch.Size([896, 336, 1]) torch.Size([896, 168, 1])   torch.Size([896, 84, 1])    torch.Size([896, 42, 1])
            x_list.append(x)

        # 3. Not Channel-Independency
        x_list = self.pre_enc(x_list)

        # 4. 不同尺度的子序列分别经过相同的Embedding, 注意这里采用的是类似于TimesNet的嵌入方式，而非PatchEmbedding嵌入方式
        # print(len(x_list))              # 4
        enc_out_list = []
        for i, x in enumerate(x_list[0]):
            # print(i)                    # 0
            # print(type(self.enc_embedding)) # <class 'models.TimeMixer.DataEmbedding_wo_pos'>
            # print(x.shape)              # torch.Size([896, 336, 1])   torch.Size([896, 168, 1])   torch.Size([896, 84, 1])   torch.Size([896, 42, 1])
            enc_out = self.enc_embedding(x, None)
            # print(enc_out.shape)        # torch.Size([896, 336, 16])  torch.Size([896, 168, 16])  torch.Size([896, 84, 16])  torch.Size([896, 42, 16])
            enc_out_list.append(enc_out)

        # 5. Past Decomposable Mixing (作为Encoder从ObservationSequence中捕获representation)
        # print(self.layer)               # 2
        for i in range(self.layer):
            # print(type(self.pdm_blocks[i])) # <class 'models.TimeMixer.PastDecomposableMixing'>
            # print(len(enc_out_list))    # 4
            # for tensor_ in enc_out_list:
            #     print(tensor_.shape)    # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
            enc_out_list = self.pdm_blocks[i](enc_out_list)
            # print(len(enc_out_list))    # 4
            # for tensor_ in enc_out_list:
            #     print(tensor_.shape)    # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])

        # 6. Future Multipredictor Mixing (作为Decoder重构ForecastingSequence)
        # print(self.channel_independence)# 1
        dec_out_list = []
        # print(len(x_list))          # 4
        if self.channel_independence:
            for i, enc_out in enumerate(enc_out_list):
                # print(i)                # 0                             1                           2                           3
                # print(type(self.predict_layers[i])) # <class 'torch.nn.modules.linear.Linear'>
                # print(type(self.projection_layer))  # <class 'torch.nn.modules.linear.Linear'>
                # print(enc_out.shape)    # torch.Size([896, 336, 16])    torch.Size([896, 168, 16])  torch.Size([896, 84, 16])   torch.Size([896, 42, 16])
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                # print(dec_out.shape)    # torch.Size([896, 96, 16])     same
                dec_out = self.projection_layer(dec_out)
                # print(dec_out.shape)    # torch.Size([896, 96, 1])      same
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                # print(dec_out.shape)    # torch.Size([128, 96, 7])      same
                dec_out_list.append(dec_out)
        else:
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, x_list[1])):
                # print(i)                # 0                             1                           2                           3

                # print(type(self.predict_layers[i]))     # <class 'torch.nn.modules.linear.Linear'>
                # print(type(self.projection_layer))      # <class 'torch.nn.modules.linear.Linear'>
                # print(enc_out.shape)    # torch.Size([32, 336, 16])     torch.Size([32, 168, 16])   torch.Size([32, 84, 16])    torch.Size([32, 42, 16])
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                # print(dec_out.shape)    # torch.Size([32, 96, 16])
                dec_out = self.projection_layer(dec_out)
                # print(dec_out.shape)    # torch.Size([32, 96, 7])

                # print(type(self.out_res_layers[i]))     # <class 'torch.nn.modules.linear.Linear'>
                # print(type(self.regression_layers[i]))  # <class 'torch.nn.modules.linear.Linear'>
                # print(out_res.shape)    # torch.Size([32, 336, 7])      torch.Size([32, 168, 7])    torch.Size([32, 84, 7])     torch.Size([32, 42, 7])
                out_res = out_res.permute(0, 2, 1)
                # print(out_res.shape)    # torch.Size([32, 7, 336])      torch.Size([32, 7, 168])    torch.Size([32, 7, 84])     torch.Size([32, 7, 42])
                out_res = self.out_res_layers[i](out_res)
                # print(out_res.shape)    # torch.Size([32, 7, 336])      torch.Size([32, 7, 168])    torch.Size([32, 7, 84])     torch.Size([32, 7, 42])
                out_res = self.regression_layers[i](out_res)
                # print(out_res.shape)    # torch.Size([32, 7, 96])
                out_res = out_res.permute(0, 2, 1)
                # print(out_res.shape)    # torch.Size([32, 96, 7])

                dec_out = dec_out + out_res
                # print(dec_out.shape)    # torch.Size([32, 96, 7])
                dec_out_list.append(dec_out)

        # 7. Multi-scale Fusion
        dec_out = torch.stack(dec_out_list, dim=-1)
        # print(dec_out.shape)            # torch.Size([128, 96, 7, 4])
        dec_out = dec_out.sum(-1)
        # print(dec_out.shape)            # torch.Size([128, 96, 7])

        # 8. DeNormalization
        # print(type(self.normalize_layers[0]))# <class 'models.TimeMixer.Normalize'>
        # print(dec_out.shape)            # torch.Size([128, 96, 7])
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        # print(dec_out.shape)            # torch.Size([128, 96, 7])
        # print('############# TimeMixer.forecast-2')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-Criss CrossAttention as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def anomaly_detection(self, x_enc):
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # print(x_enc.shape)      # torch.Size([16, 1751, 3])
        # print(x_mark_enc.shape) # torch.Size([16, 1751])
        # print((x_mark_enc == 1.0).all())    # True
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # print(len(enc_out_list))    # 4
        # for tensor_ in enc_out_list:
        #     print(tensor_.shape)    # torch.Size([48, 1751, 32])    torch.Size([48, 875, 32])    torch.Size([48, 437, 32])    torch.Size([48, 218, 32])

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        else:
            raise ValueError('Other tasks implemented yet')


if __name__ == '__main__':
    # bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
    print('Hello, World!')
