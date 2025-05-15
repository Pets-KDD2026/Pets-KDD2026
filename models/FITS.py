import torch
import torch.nn as nn

from adapter_modules.comer_modules import Normalize


# FITS: Frequency Interpolation Time Series Forecasting
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = False
        self.channels = configs.enc_in

        self.dominance_freq = configs.cut_freq  # 720/24
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        self.normalize_layer = Normalize(self.feature_size, affine=True, non_norm=False)

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)

        # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len

    def forecast(self, x):
        x = self.normalize_layer(x, 'norm')

        print(x.shape)
        low_specx = torch.fft.rfft(x, dim=1)
        print(low_specx.shape)
        low_specx[:, self.dominance_freq:] = 0
        print(low_specx.shape)
        low_specx = low_specx[:, 0:self.dominance_freq, :]
        print(low_specx.shape)

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            print(type(self.freq_upsampler))
            print(low_specx.shape)
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)
            print(low_specxy_.shape)

        low_specxy = torch.zeros([low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)], dtype=low_specxy_.dtype).to(low_specxy_.device)
        print(low_specxy.shape)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        print(low_specxy.shape)
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        print(low_xy.shape)
        low_xy = low_xy * self.length_ratio
        print(low_xy.shape)
        # dom_xy=self.Dlinear(dom_x)
        xy = self.normalize_layer(low_xy, 'denorm')
        return xy

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