import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from models.TimeMixer import Model
from adapter_modules._for_TimeMixer import TMPTimeMixerEmbedding, AdapterTimeMixerBlock, DecodeHeadTimeMixer


# 1.1 MultiScaleSeasonMixing
# Bottom-up (high->low) mixing season pattern
# 输入一组多尺度季节序列: (896,16,336), (896,16,168), (896,16,84), (896,16,42)
# 对于两个相邻尺度的序列out_high(336)和out_low(168), 基于MLP_336_168将out_high下采样到长度为168, 并与out_low相加作为新的out_low，
# 随后为接下来的相邻尺度序列重复这一操作, 从而将(896,16,168),(896,16,84),(896,16,42)这三个尺度的序列都融入了更大尺度的【季节时序信息】,
# 最后将原始的(896,16,336)和更新后的(896,16,168),(896,16,84),(896,16,42)组成列表并返回

# 1.2 MultiScaleTrendMixing
# Top->down (low->high) mixing trend pattern
# 输入一组多尺度季节序列: (896,16,42), (896,16,84), (896,16,168), (896,16,336)
# 对于两个相邻尺度的序列out_low(42)和out_high(84), 基于MLP_42_84将out_low上采样到长度为84, 并与out_high相加作为新的out_high，
# 随后为接下来的相邻尺度序列重复这一操作, 从而将(896,16,84),(896,16,168),(896,16,336)这三个尺度的序列都融入了更小尺度的【趋势时序信息】,
# 最后将原始的(896,16,42)和更新后的(896,16,84),(896,16,168),(896,16,336)组成列表并返回

# 2.1 Adapter Embedding:
#  x1(896,336,1),  x2(896,168,1),  x3(896,84,1),  x4(896,42,1) -TMPQ->
# x11(896,336,1), x21(896,168,1), x31(896,84,1), x41(896,42,1) +
# x12(896,336,1), x22(896,168,1), x32(896,84,1), x42(896,42,1) +
# x13(896,336,1), x23(896,168,1), x33(896,84,1), x43(896,42,1) -Embedding->
# e11(896,336,16),e21(896,168,16),e31(896,84,16),e41(896,42,16) +
# e12(896,336,16),e22(896,168,16),e32(896,84,16),e42(896,42,16) +
# e13(896,336,16),e23(896,168,16),e33(896,84,16),e43(896,42,16) -Concat->
#  c1(896,336,48), c2(896,168,48), c3(896,84,48), c4(896,42,48)

# 2.2 Adapter Encoder
# CrossLayer_down_1: c1(896,336,48) -> c1(896,336,16)
# CrossLayer_down_2: c2(896,168,48) -> c2(896,168,16)
# CrossLayer_down_3: c3(896,84,48)  -> c3(896,84,16)
# CrossLayer_down_4: c4(896,42,48)  -> c4(896,42,16)
# CrossLayer_up_1: c1(896,336,16) -> c1(896,336,48)
# CrossLayer_up_2: c2(896,168,16) -> c2(896,168,48)
# CrossLayer_up_3: c3(896,84,16)  -> c3(896,84,48)
# CrossLayer_up_4: c4(896,42,16)  -> c4(896,42,48)

# 3. PastDecomposableMixing as Encoder
# 输入一层多尺度原始序列, x1(896,336,16), x2(896,168,16), x3(896,84,16), x4(896,42,16)
#      季节趋势分解得到, s1(896,336,16), s2(896,168,16), s3(896,84,16), s4(896,42,16)
#      季节趋势分解得到, t1(896,336,16), t2(896,168,16), t3(896,84,16), t4(896,42,16)
#      在(s1,s2,s3,s4)之间融合信息建立连接, 在(t1,t2,t3,t4)之间融合信息建立连接,
#      残差连接计算, o1=x1+MLP(s1+t1), o2=x2+MLP(s2+t2), o3=x3+MLP(s3+t3), o4=x4+MLP(s4+t4),
# 输出一层多尺度原始序列, o1(896,336,16), o2(896,168,16), o3(896,84,16), o4(896,42,16)

# 【OverAll】
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


# 将整数m划分为n个整数，使得到的整数之间的差值不超过1，结果按照升序排列。
def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n


class ModelAdapter(Model):
    def __init__(
            self,
            configs,                                        # Backbone独有
            period_num=3,                                   # Adapter独有
    ):
        super().__init__(configs=configs)
        # print('############# TimeMixer_Adapter-1')
        # ######################## Original Module
        # print(self.task_name)                   # long_term_forecast
        # print(self.seq_len)                     # 336
        # print(self.pred_len)                    # 96
        # print(self.down_sampling_window)        # 2
        # print(self.channel_independence)        # 1

        # print(self.configs.e_layers)            # 4
        # print(len(self.pdm_blocks))             # 4
        # print(print(type(self.pdm_blocks[0])))  # <class 'models.TimeMixer.PastDecomposableMixing'>

        # print(type(self.preprocess))            # <class 'models.TimeMixer.series_decomp'>

        # print(self.channel_independence)        # 1
        # print(type(self.enc_embedding))         # <class 'models.TimeMixer.DataEmbedding_wo_pos'>

        # print(self.configs.enc_in)              # 7
        # print(self.configs.d_model)             # 16
        # print(self.configs.embed)               # timeF
        # print(self.configs.freq)                # h
        # print(self.configs.dropout)             # 0.1

        # print(self.configs.down_sampling_layers)# 3
        # print(self.configs.enc_in)              # 7
        # print(self.configs.use_norm)            # 1
        # print(len(self.normalize_layers))       # 4
        # print(type(self.normalize_layers[0]))   # <class 'models.TimeMixer.Normalize'>
        self.channel_independence = 0 if self.task_name == 'classification' else 1

        self.adapter_layer_num = period_num+1
        self.backbone_layer_num = len(self.pdm_blocks)
        assert self.backbone_layer_num >= self.adapter_layer_num, f"The Layer Num of Backbone ({self.backbone_layer_num}) is less than Adapter ({self.adapter_layer_num})"
        split_index_list = split_integer(self.backbone_layer_num, self.adapter_layer_num)
        self.interaction_indexes = [[sum(split_index_list[:i]), sum(split_index_list[:i+1])-1] for i in range(self.adapter_layer_num)]
        # print(self.backbone_layer_num)          # 4
        # print(self.adapter_layer_num)           # 4
        # print(split_index_list)                 # [1, 1, 1, 1]
        # print(self.interaction_indexes)         # [[0, 0], [1, 1], [2, 2], [3, 3]]

        # 1. SPM, 用于生成multi-scale的c
        self.spm = TMPTimeMixerEmbedding(
            d_model=self.configs.d_model, embed=self.configs.embed, freq=self.configs.freq, dropout=self.configs.dropout,
            channel_independence=self.channel_independence, enc_in=self.configs.enc_in,
        )
        self.spm.apply(self._init_weights)
        # print(type(self.spm))                   # <class 'adapter_modules._for_TimeMixer.TMPTimeMixerEmbedding'>

        # 2. Multi-Level Embedding, 用于为multi-scale的c添加层级嵌入信息
        self.level_embed = nn.Parameter(torch.zeros(period_num+1, self.configs.d_model))
        normal_(self.level_embed)
        # print(self.level_embed.shape)           # torch.Size([4, 16])

        # 3. 基于BackboneBlock封装得到的AdapterBlock, 其中负责将c和x融合，并
        self.interactions = nn.Sequential(*[
            AdapterTimeMixerBlock(d_model=self.configs.d_model, cond_num=self.configs.down_sampling_layers+1)
            for _ in range(len(self.interaction_indexes))
        ])
        self.interactions.apply(self._init_weights)

        # 4. Decode Head
        # self.norm_layers = torch.nn.ModuleList([
        #     nn.LayerNorm(self.configs.d_model * 4) for _ in range(self.configs.down_sampling_layers + 1)
        # ])
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #     self.head_ppm = DecodeHeadTimeMixer(
        #         token_num_max=self.seq_len,
        #         d_model=self.configs.d_model * 4,
        #         pred_len=self.pred_len,
        #         period_num=self.configs.down_sampling_layers+1,
        #     )
        # elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        #     self.head_ppm = DecodeHeadTimeMixer(
        #         token_num_max=self.seq_len,
        #         d_model=self.configs.d_model * 4,
        #         pred_len=self.seq_len,
        #         period_num=self.configs.down_sampling_layers + 1,
        #     )
        # elif self.task_name == 'classification':
        #     self.projection = nn.Linear(
        #         in_features=(self.token_num_1 + self.token_num_2 * 2 + self.token_num_3) * self.d_model * self.enc_in,
        #         out_features=configs.num_class
        #     )

        # 4. Decode Head
        self.norm_layers = torch.nn.ModuleList([
            nn.LayerNorm(self.configs.d_model) for _ in range(self.configs.down_sampling_layers + 1)
        ])
        if self.channel_independence:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # print('############# TimeMixer_Adapter-2')

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

        # 3.0 Not Channel-Independency
        x_list = self.pre_enc(x_list)

        # 3.1 SPM forward
        # 3.2 Multi-Level Embedding
        c_list = self.spm(x_list=x_list[0], level_embed=self.level_embed)
        # print(len(c_list))              # 4
        # for tensor_ in c_list:
        #     print(tensor_.shape)        # torch.Size([896, 336, 64])  torch.Size([896, 168, 64])  torch.Size([896, 84, 64])  torch.Size([896, 42, 64])

        # 4. 不同尺度的子序列分别经过相同的Embedding, 注意这里采用的是类似于TimesNet的嵌入方式，而非PatchEmbedding嵌入方式
        # print(len(x_list[0))            # 4
        emb_out_list = []
        for i, x in enumerate(x_list[0]):
            # print(i)                    # 0
            # print(type(self.enc_embedding)) # <class 'models.TimeMixer.DataEmbedding_wo_pos'>
            # print(x.shape)              # torch.Size([896, 336, 1])   torch.Size([896, 168, 1])   torch.Size([896, 84, 1])   torch.Size([896, 42, 1])
            emb_out = self.enc_embedding(x, None)
            # print(emb_out.shape)        # torch.Size([896, 336, 16])  torch.Size([896, 168, 16])  torch.Size([896, 84, 16])  torch.Size([896, 42, 16])
            emb_out_list.append(emb_out)

        # print(len(self.interactions))   # 4
        # print(len(self.pdm_blocks))     # 12
        # outs_2d = []
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.pdm_blocks[indexes[0]:indexes[-1] + 1]
            emb_out_list, c_list = adapter_block(x_list=emb_out_list, c_list=c_list, backbone_blocks=backbone_blocks)
            # outs_2d.append(emb_out_list)

        # fusion_outs_2d = []
        # for i, outs_1d in enumerate(outs_2d):
        #     fusion_outs_1d = []
        #     for j, outs in enumerate(outs_1d):
        #         if i == j:
        #             fusion_outs_1d.append(outs)
        #         else:
        #             outs = F.interpolate(
        #                 outs.transpose(1, 2).contiguous(),
        #                 scale_factor=outs_1d[i].shape[1] / outs.shape[1],
        #                 mode='linear', align_corners=False, recompute_scale_factor=True
        #             ).transpose(1, 2).contiguous()
        #             fusion_outs_1d.append(outs)
        #     fusion_outs_1d = torch.concat(fusion_outs_1d, dim=2)
        #     fusion_outs_2d.append(fusion_outs_1d)
        # print(len(fusion_outs_2d))      # 4
        # for tensor_ in fusion_outs_2d:
        #     print(tensor_.shape)        # torch.Size([896, 336, 64])  torch.Size([896, 168, 64])  torch.Size([896, 84, 64])  torch.Size([896, 42, 64])

        enc_out_list = []
        for i in range(len(emb_out_list)):
            enc_out = self.norm_layers[i](emb_out_list[i])
            enc_out_list.append(enc_out)
        # print(len(enc_out_list))        # 4
        # for tensor_ in enc_out_list:
        #     print(tensor_.shape)        # torch.Size([896, 336, 64])  torch.Size([896, 168, 64])  torch.Size([896, 84, 64])  torch.Size([896, 42, 64])

        """
        dec_out = self.head_ppm(enc_out_list)
        # print(dec_out.shape)            # torch.Size([896, 96])
        dec_out = dec_out.reshape(B, N, dec_out.shape[-1]).permute(0, 2, 1).contiguous()
        # print(dec_out.shape)            # torch.Size([128, 96, 7])
        """

        # 6. Future Multipredictor Mixing (作为Decoder重构ForecastingSequence)
        # print(self.channel_independence)# 1
        dec_out_list = []
        # print(len(x_list))          # 4
        if self.channel_independence:
            for i, enc_out in enumerate(enc_out_list):
                # print(i)                # 0                             1                           2                           3
                # print(type(self.predict_layers[i])) # <class 'torch.nn.modules.linear.Linear'>
                # print(type(self.projection_layer))  # <class 'torch.nn.modules.linear.Linear'>
                # print(enc_out.shape)    # torch.Size([896, 336, 64])    torch.Size([896, 168, 64])  torch.Size([896, 84, 64])   torch.Size([896, 42, 64])
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                # print(dec_out.shape)    # torch.Size([896, 96, 64])     same
                dec_out = self.projection_layer(dec_out)
                # print(dec_out.shape)    # torch.Size([896, 96, 1])      same
                dec_out = dec_out.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()
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
        # 1. 首先将长度为336的原始序列拆解多尺度分量336,168,84,42, 有点类似于多层的离散小波变换
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # 2. 基于RevIn的Normalization预处理
        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # 3.1 Not Channel-Independency
        x_list = self.pre_enc(x_list)

        # 3.2 SPM forward
        c_list = self.spm(x_list=x_list[0], level_embed=self.level_embed)

        # 4. 不同尺度的子序列分别经过相同的Embedding, 注意这里采用的是类似于TimesNet的嵌入方式，而非PatchEmbedding嵌入方式
        emb_out_list = []
        for i, x in enumerate(x_list[0]):
            emb_out = self.enc_embedding(x, None)
            emb_out_list.append(emb_out)

        # 5. Encoder
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.pdm_blocks[indexes[0]:indexes[-1] + 1]
            emb_out_list, c_list = adapter_block(x_list=emb_out_list, c_list=c_list, backbone_blocks=backbone_blocks)
        enc_out_list = []
        for i in range(len(emb_out_list)):
            enc_out = self.norm_layers[i](emb_out_list[i])
            enc_out_list.append(enc_out)

        # 6. Output (不同于forecast)
        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def anomaly_detection(self, x_enc):
        # 1. 首先将长度为336的原始序列拆解多尺度分量336,168,84,42, 有点类似于多层的离散小波变换
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # 2. 基于RevIn的Normalization预处理
        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # 3.1 Not Channel-Independency
        x_list = self.pre_enc(x_list)

        # 3.2 SPM forward
        c_list = self.spm(x_list=x_list[0], level_embed=self.level_embed)

        # 4. 不同尺度的子序列分别经过相同的Embedding, 注意这里采用的是类似于TimesNet的嵌入方式，而非PatchEmbedding嵌入方式
        emb_out_list = []
        for i, x in enumerate(x_list[0]):
            emb_out = self.enc_embedding(x, None)
            emb_out_list.append(emb_out)

        # 5. Encoder
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.pdm_blocks[indexes[0]:indexes[-1] + 1]
            emb_out_list, c_list = adapter_block(x_list=emb_out_list, c_list=c_list, backbone_blocks=backbone_blocks)
        enc_out_list = []
        for i in range(len(emb_out_list)):
            enc_out = self.norm_layers[i](emb_out_list[i])
            enc_out_list.append(enc_out)

        # 6. Output (不同于forecast)
        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # 1. 首先将长度为336的原始序列拆解多尺度分量336,168,84,42, 有点类似于多层的离散小波变换
        x_enc_list = self.__multi_scale_process_inputs(x_enc)
        # print(len(x_enc_list))      # 4
        # for tensor_ in x_enc_list:
        #     print(tensor_.shape)    # (32,1751,3)   (32,875,3)   (32,437,3)   (32,218,3)

        # 2. 基于RevIn的Normalization预处理
        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)
        # print(len(x_list))          # 4
        # for tensor_ in x_list:
        #     print(tensor_.shape)    # (32,1751,3)   (32,875,3)   (32,437,3)   (32,218,3)

        # 3.1 Not Channel-Independency
        x_list = self.pre_enc(x_list)

        # 3.2 SPM forward
        c_list = self.spm(x_list=x_list[0], level_embed=self.level_embed)
        # print(len(c_list))          # 4
        # for tensor_ in c_list:
        #     print(tensor_.shape)    # (32,1751,64)  (32,875,64)  (32,437,64)  (32,218,64)

        # 4. 不同尺度的子序列分别经过相同的Embedding, 注意这里采用的是类似于TimesNet的嵌入方式，而非PatchEmbedding嵌入方式
        emb_out_list = []
        for i, x in enumerate(x_list[0]):
            emb_out = self.enc_embedding(x, None)
            emb_out_list.append(emb_out)
        # print(len(emb_out_list))    # 4
        # for tensor_ in emb_out_list:
        #     print(tensor_.shape)    # (32,1751,16)  (32,875,16)  (32,437,16)  (32,218,16)

        # 5. Encoder
        for i in range(len(self.interactions)):
            indexes = self.interaction_indexes[i]
            adapter_block = self.interactions[i]
            backbone_blocks = self.pdm_blocks[indexes[0]:indexes[-1] + 1]
            emb_out_list, c_list = adapter_block(x_list=emb_out_list, c_list=c_list, backbone_blocks=backbone_blocks)
        enc_out_list = []
        for i in range(len(emb_out_list)):
            enc_out = self.norm_layers[i](emb_out_list[i])
            enc_out_list.append(enc_out)
        # print(len(enc_out_list))  # 4
        # for tensor_ in enc_out_list:
        #     print(tensor_.shape)  # (32,1751,16)  (32,875,16)  (32,437,16)  (32,218,16)

        # 6. Output (不同于forecast)
        output = self.act(enc_out_list[0])
        output = self.dropout(output)
        # print(output.shape)         # torch.Size([16, 1751, 16])
        # zero-out padding embeddings
        # print(x_mark_enc.shape)     # torch.Size([16, 1751])
        output = output * x_mark_enc.unsqueeze(-1)
        # print(output.shape)         # torch.Size([16, 1751, 16])
        output = output.reshape(output.shape[0], -1)
        # print(output.shape)         # torch.Size([16, 28016])
        output = self.projection(output)
        # print(output.shape)         # torch.Size([16, 4])
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
