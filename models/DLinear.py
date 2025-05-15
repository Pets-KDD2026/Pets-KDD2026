import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.hidden_len = configs.seq_len
        else:
            self.hidden_len = configs.pred_len

        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.hidden_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.hidden_len))

                self.Linear_Seasonal[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.hidden_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.hidden_len)

            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # print('############# DLinear.encoder-1')

        # print(type(self.decompsition))          # <class 'layers.Autoformer_EncDec.series_decomp'>
        # print(x.shape)                          # torch.Size([32, 336, 7])
        seasonal_init, trend_init = self.decompsition(x)
        # print(seasonal_init.shape)              # torch.Size([32, 336, 7])
        # print(trend_init.shape)                 # torch.Size([32, 336, 7])
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1).contiguous(), trend_init.permute(0, 2, 1).contiguous()
        # print(seasonal_init.shape)              # torch.Size([32, 7, 336])
        # print(trend_init.shape)                 # torch.Size([32, 7, 336])

        # print(self.individual)                  # False
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.hidden_len],
                dtype=seasonal_init.dtype
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.hidden_len],
                dtype=trend_init.dtype
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # print(type(self.Linear_Seasonal))   # <class 'torch.nn.modules.linear.Linear'>
            # print(seasonal_init.shape)          # torch.Size([32, 7, 336])
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            # print(seasonal_output.shape)        # torch.Size([32, 7, 96])

            # print(type(self.Linear_Trend))      # <class 'torch.nn.modules.linear.Linear'>
            # print(trend_init.shape)             # torch.Size([32, 7, 336])
            trend_output = self.Linear_Trend(trend_init)
            # print(trend_output.shape)           # torch.Size([32, 7, 96])

        x = seasonal_output + trend_output
        # print(x.shape)                          # torch.Size([32, 7, 96])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)                          # torch.Size([32, 96, 7])

        # print('############# DLinear.encoder-2')
        return x

    def forecast(self, x_enc):
        output = self.encoder(x_enc)
        return output

    def imputation(self, x_enc):
        output = self.encoder(x_enc)
        return output

    def anomaly_detection(self, x_enc):
        output = self.encoder(x_enc)
        return output

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1).contiguous()
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.hidden_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
