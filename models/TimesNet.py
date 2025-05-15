import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('############# Inception_Block_V1-1')

        # print(self.num_kernels)             # 6                                   6
        res_list = []
        for i in range(self.num_kernels):
            # print(i)                        # 0,1,2,3,4,5                         0,1,2,3,4,5
            tmp = self.kernels[i](x)
            # print(type(self.kernels[i]))    # <class 'torch.nn.modules.conv.Conv2d'>
            # print(x.shape)                  # torch.Size([32, 16, 48, 9])         torch.Size([32, 32, 48, 9])
            # print(tmp.shape)                # torch.Size([32, 32, 48, 9])         torch.Size([32, 16, 48, 9])
            res_list.append(tmp)
        res = torch.stack(res_list, dim=-1)
        # print(res.shape)                    # torch.Size([32, 32, 48, 9, 6])      torch.Size([32, 16, 48, 9, 6])
        res = res.mean(-1)
        # print(res.shape)                    # torch.Size([32, 32, 48, 9])         torch.Size([32, 16, 48, 9])

        # print('############# Inception_Block_V1-2')
        return res


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # print('############# TimesBlock-1')
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # print(x.shape)                  # torch.Size([32, 432, 16])
        # print(self.k)                   # 5
        # print(period_list.shape)        # (5,)
        # print(period_weight.shape)      # torch.Size([32, 5])

        res = []
        for i in range(self.k):
            # print(i)                    # 0                             1                               2                               3                               4
            period = period_list[i]
            # print(period)               # 9                             2                               2                               6                               2

            # 1. Padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # print(x.shape)              # torch.Size([32, 432, 16])
            # print(out.shape)            # torch.Size([32, 432, 16])

            # 2. Reshape
            out = out.reshape(B, length // period, period, N).contiguous().permute(0, 3, 1, 2).contiguous()
            # print(out.shape)            # torch.Size([32, 16, 48, 9])   torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 72, 6])     torch.Size([32, 16, 216, 2])

            # 3. 2D_conv_based From 1d_tensor to 2d_tensor
            # print(type(self.conv))      # <class 'torch.nn.modules.container.Sequential'>
            # print(out.shape)            # torch.Size([32, 16, 48, 9])   torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 72, 6])     torch.Size([32, 16, 216, 2])
            out = self.conv(out)
            # print(out.shape)            # torch.Size([32, 16, 48, 9])   torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 216, 2])    torch.Size([32, 16, 72, 6])     torch.Size([32, 16, 216, 2])
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B, -1, N).contiguous()
            # print(out.shape)            # torch.Size([32, 432, 16])
            out = out[:, :(self.seq_len + self.pred_len), :]
            # print(out.shape)            # torch.Size([32, 432, 16])
            res.append(out)

        res = torch.stack(res, dim=-1)
        # print(res.shape)                # torch.Size([32, 432, 16, 5])

        # 4. Adaptive Aggregation
        # print(period_weight.shape)      # torch.Size([32, 5])
        period_weight = F.softmax(period_weight, dim=1)
        # print(period_weight.shape)      # torch.Size([32, 5])
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        # print(period_weight.shape)      # torch.Size([32, 432, 16, 5])
        res = torch.sum(res * period_weight, -1)
        # print(res.shape)                # torch.Size([32, 432, 16])

        res = res + x
        # print('############# TimesBlock-2')
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print('############# TimesNet.forecast-1')
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 1. Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # print(type(self.enc_embedding))     # <class 'layers.Embed.DataEmbedding'>
        # print(x_enc.shape)                  # torch.Size([32, 336, 7])
        # print(x_mark_enc)                   # None
        # print(enc_out.shape)                # torch.Size([32, 336, 16])

        enc_out = self.predict_linear(enc_out.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # print(type(self.predict_linear))    # <class 'torch.nn.modules.linear.Linear'>
        # print(enc_out.shape)                # torch.Size([32, 432, 16])

        # 2. TimesNet Backbone
        # print(self.layer)                   # 2
        for i in range(self.layer):
            # print(i)                        # 0
            layer = self.model[i]
            # print(type(layer))              # <class 'models.TimesNet.TimesBlock'>
            # print(type(self.layer_norm))    # <class 'torch.nn.modules.normalization.LayerNorm'>
            # print(enc_out.shape)            # torch.Size([32, 432, 16])
            enc_out = self.layer_norm(layer(enc_out))
            # print(enc_out.shape)            # torch.Size([32, 432, 16])

        # 3. Projection
        dec_out = self.projection(enc_out)
        # print(type(self.projection))        # <class 'torch.nn.modules.linear.Linear'>
        # print(enc_out.shape)                # torch.Size([32, 432, 16])
        # print(dec_out.shape)                # torch.Size([32, 432, 7])

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        # print('############# TimesNet.forecast-2')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1).contiguous()
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
