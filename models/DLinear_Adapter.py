import torch
import torch.nn as nn

from adapter_modules.comer_modules import Normalize
from adapter_modules.trend_multi_period_quantized_wavelet import TMPQ


class ModelAdapter(nn.Module):
    def __init__(self, configs, individual=False):
        super(ModelAdapter, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.hidden_len = configs.seq_len
        else:
            self.hidden_len = configs.pred_len

        self.individual = individual
        self.channels = configs.enc_in

        # ReViN, Normalization
        self.normalize_layer = Normalize(self.channels, affine=True, non_norm=False)

        if self.individual:
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Seasonal_1 = nn.ModuleList()
            self.Linear_Seasonal_2 = nn.ModuleList()
            self.Linear_Seasonal_3 = nn.ModuleList()
            self.Linear_Residual = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.hidden_len))
                self.Linear_Seasonal_1.append(nn.Linear(self.seq_len, self.hidden_len))
                self.Linear_Seasonal_2.append(nn.Linear(self.seq_len, self.hidden_len))
                self.Linear_Seasonal_3.append(nn.Linear(self.seq_len, self.hidden_len))
                self.Linear_Residual.append(nn.Linear(self.seq_len, self.hidden_len))

                self.Linear_Trend[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
                self.Linear_Seasonal_1[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
                self.Linear_Seasonal_2[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
                self.Linear_Seasonal_3[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
                self.Linear_Residual[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.hidden_len)
            self.Linear_Seasonal_1 = nn.Linear(self.seq_len, self.hidden_len)
            self.Linear_Seasonal_2 = nn.Linear(self.seq_len, self.hidden_len)
            self.Linear_Seasonal_3 = nn.Linear(self.seq_len, self.hidden_len)
            self.Linear_Residual = nn.Linear(self.seq_len, self.hidden_len)

            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
            self.Linear_Seasonal_1.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
            self.Linear_Seasonal_2.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
            self.Linear_Seasonal_3.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))
            self.Linear_Residual.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.hidden_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x_enc):
        # print('############# DLinear.encoder-1')

        # print(x_enc.shape)                      # torch.Size([32, 336, 7])
        x_enc = self.normalize_layer(x_enc, 'norm')
        # print(x_enc.shape)                      # torch.Size([32, 336, 7])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                      # torch.Size([32, 7, 336])

        # 1. Trend Multi-Period Quantized
        tmpq_dict = TMPQ(x_enc)
        init_t = tmpq_dict['trend'][:, :, :x_enc.shape[-1]]
        init_s_1 = tmpq_dict['seasonal_1'][:, :, :x_enc.shape[-1]]
        init_s_2 = tmpq_dict['seasonal_2'][:, :, :x_enc.shape[-1]]
        init_s_3 = tmpq_dict['seasonal_3'][:, :, :x_enc.shape[-1]]
        # print(init_t.shape)                     # torch.Size([32, 7, 336])
        # print(init_s_1.shape)                   # torch.Size([32, 7, 336])
        # print(init_s_2.shape)                   # torch.Size([32, 7, 336])
        # print(init_s_3.shape)                   # torch.Size([32, 7, 336])

        # 2. Linear-based Backbone
        # print(self.individual)                  # False
        if self.individual:
            output_t = torch.zeros([init_t.size(0), init_t.size(1), self.hidden_len], dtype=init_t.dtype).to(init_t.device)
            output_s_1 = torch.zeros([init_s_1.size(0), init_s_1.size(1), self.hidden_len], dtype=init_s_1.dtype).to(init_s_1.device)
            output_s_2 = torch.zeros([init_s_2.size(0), init_s_2.size(1), self.hidden_len], dtype=init_s_2.dtype).to(init_s_2.device)
            output_s_3 = torch.zeros([init_s_3.size(0), init_s_3.size(1), self.hidden_len], dtype=init_s_3.dtype).to(init_s_3.device)
            for i in range(self.channels):
                output_t[:, i, :] = self.Linear_Trend[i](init_t[:, i, :])
                output_s_1[:, i, :] = self.Linear_Seasonal_1[i](init_s_1[:, i, :])
                output_s_2[:, i, :] = self.Linear_Seasonal_2[i](init_s_2[:, i, :])
                output_s_3[:, i, :] = self.Linear_Seasonal_3[i](init_s_3[:, i, :])
        else:
            output_t = self.Linear_Trend(init_t)
            output_s_1 = self.Linear_Seasonal_1(init_s_1)
            output_s_2 = self.Linear_Seasonal_2(init_s_2)
            output_s_3 = self.Linear_Seasonal_3(init_s_3)

        # 3. Fusion
        x = output_t + output_s_1 + output_s_2 + output_s_3
        # print(x.shape)                          # torch.Size([32, 7, 96])
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)                          # torch.Size([32, 96, 7])
        x = self.normalize_layer(x, 'denorm')
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
