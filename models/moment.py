import logging
import warnings
from argparse import Namespace
from copy import deepcopy
from math import ceil
from dataclasses import dataclass
import torch
import numpy.typing as npt
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import T5Config, T5EncoderModel, T5Model

from models.moment_module import RevIN, Masking, PatchEmbedding, Patching


def model_show(model):
    show = True

    if show:
        print('#########################################################')

        print('')
        print('With Grad Parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        print('')
        print('Without Grad Parameters:')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(name)

        print('#########################################################')
    else:
        return None


def get_anomaly_criterion(anomaly_criterion: str = "mse"):
    if anomaly_criterion == "mse":
        return torch.nn.MSELoss(reduction="none")
    elif anomaly_criterion == "mae":
        return torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Anomaly criterion {anomaly_criterion} not supported.")


def freeze_parameters(model):
    """
    Freeze parameters of the model
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model


class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()
        for attr in dir(namespace):
            if not attr.startswith("__"):
                setattr(new_instance, attr, getattr(namespace, attr))
        return new_instance

    def getattr(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class TASKS:
    RECONSTRUCTION: str = "reconstruction"
    FORECASTING: str = "forecasting"
    CLASSIFICATION: str = "classification"
    EMBED: str = "embedding"


@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    logits: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False


SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class PretrainHead(nn.Module):
    def __init__(self, d_model=768, patch_len=8, head_dropout=0.1, orth_gain=1.41):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        # print('############# PretrainHead-1')
        # print(x.shape)                      # torch.Size([224, 1, 64, 1024])
        x = self.linear(self.dropout(x))
        # print(x.shape)                      # torch.Size([224, 1, 64, 8])
        x = x.flatten(start_dim=2, end_dim=3)
        # print(x.shape)                      # torch.Size([224, 1, 512])
        # print('############# PretrainHead-2')
        return x


class ClassificationHead(nn.Module):
    def __init__(self, n_channels=1, d_model=768, n_classes=2, head_dropout=0.1, reduction="concat"):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        if reduction == "mean":
            self.linear = nn.Linear(d_model, n_classes)
        elif reduction == "concat":
            self.linear = nn.Linear(n_channels * d_model, n_classes)
        else:
            raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")

    def forward(self, x):
        # print('############# ClassificationHead-1')
        # print(x.shape)                      # torch.Size([32, 64, 1024])
        x = torch.mean(x, dim=1)
        # print(x.shape)                      # torch.Size([32, 1024])
        x = self.dropout(x)
        # print(x.shape)                      # torch.Size([32, 1024])
        y = self.linear(x)
        # print(y.shape)                      # torch.Size([32, 5])
        # print('############# ClassificationHead-2')
        return y


class ForecastingHead(nn.Module):
    def __init__(self, head_nf=768*64, pred_len=96, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, pred_len)

    def forward(self, x):
        # print('############# ForecastingHead-1')
        # print(x.shape)                      # torch.Size([32, 7, 64, 1024])
        x = self.flatten(x)
        # print(x.shape)                      # torch.Size([32, 7, 65536])
        x = self.linear(x)
        # print(x.shape)                      # torch.Size([32, 7, 192])
        x = self.dropout(x)
        # print(x.shape)                      # torch.Size([32, 7, 192])
        # print('############# ForecastingHead-2')
        return x


class Model(nn.Module):
    def __init__(self, config, adapter_task_name, pred_len, n_channels, num_class, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = adapter_task_name
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.num_class = num_class
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        # ##################################### 无需训练
        # 1. 可逆实例规范化Reversible Instance Normalization, ReIN
        #    时序预测往往面临数据分布漂移(Distribution Shift Problem)问题的挑战，例如数据的均值方差会随着时间变化而变化，这也被称为非平稳数据(Non-stationary)
        #    在时序预测任务中，训练集和测试集往往是依照时间划分的，这天然地导致了训练集和测试集之间数据分布不一致的问题，此外，不同输入序列也会存在数据分布不一致的问题
        #    为了解决上述问题，RevIN提出了一种数据规范化的方法，具体地来说，在数据输入模型前，将数据进行规范化，然后经过模型学习后得到模型输出，最后对模型输出进行反规范化
        #    RevIN是一种灵活的，端到端的可训练层，能够被应用到任意模型层
        self.normalizer = RevIN(num_features=1, affine=config.getattr("revin_affine", False))
        # 2. PatchTokenizer
        #    其中padding=0, patch_len=stride_len, 这只是简单地将时间序列折叠为若干个patch,
        self.tokenizer = Patching(patch_len=config.patch_len, stride=config.patch_stride_len)
        # 3. MaskGenerator
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))

        # ##################################### 需要训练
        # 1. PatchEmbedding
        #    在通过PatchTokenizer得到若干组patch后, 使用这个模块计算每个patch的嵌入向量
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            patch_dropout=config.getattr("patch_dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        # 2. EncoderOnly Backbone
        self.encoder = self._get_transformer_backbone(config)
        # 3. Output Head
        self.head = self._get_head('pretrain')

        # 冻结部分参数
        # 1. 对于支持Zero-shot和Finetune的Imputation,AnomalyDetection任务, 其Zero-shot期间冻结PatchEmbedding和Backbone
        # 2. 对于仅支持Finetune的Forecasting,Classification任务, 其Finetune期间冻结PatchEmbedding和Backbone, 训练OutputHead
        self.freeze_embedder = True
        self.freeze_encoder = True
        self.freeze_head = False
        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)

    def _update_inputs(self, config, **kwargs: dict) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (config.d_model is None) and (config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS):
            config.d_model = config.t5_config['d_model']
            logging.info(f"Setting d_model to {config.d_model}")
        elif config.d_model is None:
            raise ValueError("d_model must be specified if transformer backbone unless transformer backbone is a Huggingface model.")
        if config.transformer_type not in ["encoder_only", "decoder_only", "encoder_decoder"]:
            raise ValueError("transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']")
        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name in ['pretrain', 'imputation', 'anomaly_detection']:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("head_dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        elif task_name in ['classification']:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction=self.config.getattr("reduction", "concat"),
            )
        elif task_name in ['long_term_forecast', 'short_term_forecast']:
            num_patches = (max(self.config.seq_len, self.config.patch_len) - self.config.patch_len) // self.config.patch_stride_len + 1
            self.head_nf = self.config.d_model * num_patches
            return ForecastingHead(
                self.head_nf,
                self.pred_len,
                self.config.getattr("head_dropout", 0.1),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        # print('############# get_transformer_backbone-1')
        model_config = T5Config.from_dict(config.t5_config)
        # print(model_config)
        """
        # T5Config {
        #   "architectures": [
        #     "T5ForConditionalGeneration"
        #   ],
        #   "classifier_dropout": 0.0,
        #   "d_ff": 2816,
        #   "d_kv": 64,
        #   "d_model": 1024,
        #   "decoder_start_token_id": 0,
        #   "dense_act_fn": "gelu_new",
        #   "dropout_rate": 0.1,
        #   "eos_token_id": 1,
        #   "feed_forward_proj": "gated-gelu",
        #   "initializer_factor": 1.0,
        #   "is_encoder_decoder": true,
        #   "is_gated_act": true,
        #   "layer_norm_epsilon": 1e-06,
        #   "model_type": "t5",
        #   "n_positions": 512,
        #   "num_decoder_layers": 24,
        #   "num_heads": 16,
        #   "num_layers": 24,
        #   "output_past": true,
        #   "pad_token_id": 0,
        #   "relative_attention_max_distance": 128,
        #   "relative_attention_num_buckets": 32,
        #   "tie_word_embeddings": false,
        #   "transformers_version": "4.33.3",
        #   "use_cache": true,
        #   "vocab_size": 32128
        # }
        """

        # 1. 根据超参【randomly_initialize_backbone】决定是【随机初始化模型】/【加载预训练模型】
        # print(config.getattr("randomly_initialize_backbone", False))        # False
        if config.getattr("randomly_initialize_backbone", False):
            transformer_backbone = T5Model(model_config)
            logging.info(f"Initializing randomly initialized transformer from {config.transformer_backbone}.")
        else:
            transformer_backbone = T5EncoderModel(model_config)
            logging.info(f"Initializing pre-trained transformer from {config.transformer_backbone}.")

        # 2. 获取T5Model/T5EncoderModel的encoder【T5Stack_Adapter】作为backbone
        # print(type(transformer_backbone))   # <class 'transformers.models.t5.modeling_t5.T5EncoderModel'>
        transformer_backbone = transformer_backbone.get_encoder()
        # print(type(transformer_backbone))   # <class 'transformers.models.t5.modeling_t5.T5Stack'>

        # 3. 根据超参【randomly_initialize_backbone】决定是【是否允许训练backbone】
        # print(config.getattr("enable_gradient_checkpointing", True))        # True
        if config.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        # print('############# get_transformer_backbone-2')
        return transformer_backbone

    # outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
    # outputs = self.backbone_forward(inputs_embeds=enc_in, attention_mask=attention_mask)
    def backbone_forward(self, inputs_embeds, attention_mask):
        from torch.utils.checkpoint import checkpoint
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

        encoder_hidden_states = None
        use_cache = self.encoder.config.use_cache
        output_attentions = self.encoder.config.output_attentions
        # print(use_cache)                            # False
        # print(output_attentions)                    # False

        input_shape = inputs_embeds.size()[:-1]
        # print(inputs_embeds.shape)                  # torch.Size([224, 64, 1024])
        # print(input_shape)                          # torch.Size([224, 64])
        hidden_states = self.encoder.dropout(inputs_embeds)
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])

        # print(attention_mask.shape)                 # torch.Size([224, 64])
        # print(input_shape)                          # torch.Size([224, 64])
        extended_attention_mask = self.encoder.get_extended_attention_mask(attention_mask, input_shape)
        encoder_extended_attention_mask = None
        # print(extended_attention_mask.shape)        # torch.Size([224, 1, 1, 64])

        # print(self.encoder.gradient_checkpointing)  # True
        # print(self.encoder.training)                # True

        # Prepare head mask if needed
        # print(self.encoder.config.num_layers)       # 24
        head_mask = self.encoder.get_head_mask(None, self.encoder.config.num_layers)
        cross_attn_head_mask = self.encoder.get_head_mask(None, self.encoder.config.num_layers)
        # print(len(head_mask))                       # 24
        # print(head_mask[0])                         # None
        # print(len(cross_attn_head_mask))            # 24
        # print(cross_attn_head_mask[0])              # None

        # 初始化
        # print(len(self.encoder.block))              # 24
        position_bias = None
        encoder_decoder_position_bias = None
        for i, layer_module in enumerate(self.encoder.block):
            # print('')

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if self.encoder.gradient_checkpointing and self.encoder.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))
                    return custom_forward
                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=None,
                    use_cache=False,
                    output_attentions=False,
                )
            # print(i)                                # 0
            # print(type(layer_module))               # <class 'transformers.models.t5.modeling_t5.T5Block'>
            # print(hidden_states.shape)              # torch.Size([224, 64, 1024])
            # print(extended_attention_mask.shape)    # torch.Size([224, 1, 1, 64])
            # print(position_bias if i == 0 else
            #       position_bias.shape)              # None
            # print(encoder_hidden_states)            # None
            # print(encoder_extended_attention_mask)  # None
            # print(encoder_decoder_position_bias)    # None
            # print(layer_head_mask)                  # None
            # print(cross_attn_layer_head_mask)       # None

            # print(len(layer_outputs))               # 2
            hidden_states, position_bias = layer_outputs[:2]
            # print(hidden_states.shape)              # torch.Size([224, 64, 1024])
            # print(position_bias.shape)              # torch.Size([224, 16, 64, 64])

            # print(self.encoder.is_decoder)          # False
            # print(encoder_hidden_states)            # None
            if self.encoder.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3]

        # print(type(self.encoder.final_layer_norm))  # <class 'transformers.models.t5.modeling_t5.T5LayerNorm'>
        # print(type(self.encoder.dropout))           # <class 'torch.nn.modules.dropout.Dropout'>
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])
        hidden_states = self.encoder.final_layer_norm(hidden_states)
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])
        hidden_states = self.encoder.dropout(hidden_states)
        # print(hidden_states.shape)                  # torch.Size([224, 64, 1024])

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def forecast(self, x_enc, input_mask):
        # print('############# Long Forecasting-1')
        batch_size, seq_len, n_channels = x_enc.shape

        # 1. 基于RevIN实现历史序列的【可逆实例规范化】
        # (batch_size, n_channels, seq_len)
        # print(type(self.normalizer))        # <class 'models.moment_module.RevIN'>
        # print(x_enc.shape)                  # torch.Size([32, 512, 7])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        # print(input_mask.shape)             # torch.Size([32, 512])
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])

        # 2. 基于一种无需训练的方式, 直接将长度为512的历史序列拆分为64个patch, 每个patch长度为8
        # (batch_size, n_channels, seq_len) -> (batch_size, n_channels, patch_num, patch_len)
        # print(type(self.tokenizer))         # <class 'models.moment_module.Patching'>
        # print(self.config.patch_len)        # 8
        # print(self.config.patch_stride_len) # 8
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        x_enc = self.tokenizer(x=x_enc)
        # print(x_enc.shape)                  # torch.Size([32, 7, 64, 8])

        # 3. 将patch_len(8)映射为d_model(1024),
        # (batch_size, n_channels, patch_num, patch_len) -> (batch_size, n_channels, patch_num, d_model)
        # print(type(self.patch_embedding))   # <class 'models.moment_module.PatchEmbedding'>
        # print(x_enc.shape)                  # torch.Size([32, 7, 64, 8])
        # print(input_mask.shape)             # torch.Size([32, 512])
        mask = torch.ones_like(input_mask)
        # print(mask.shape)                   # torch.Size([32, 512])
        enc_in = self.patch_embedding(x_enc, mask=mask)
        # print(enc_in.shape)                 # torch.Size([32, 7, 64, 1024])

        # ChannelIndependence
        # (batch_size, n_channels, patch_num, d_model) -> (batch_size*n_channels, patch_num, d_model)
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model)).contiguous()
        # print(enc_in.shape)                 # torch.Size([224, 64, 1024])

        # 4. 为Encoder架构生成attention_mask, 其中每个样本的每个token对应一个值, 通道间共有mask值
        # (batch_size*n_channels, patch_num)
        # print(input_mask.shape)             # torch.Size([32, 512])
        # print(self.patch_len)               # 8
        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        # print(patch_view_mask.shape)        # torch.Size([32, 64])
        # print(n_channels)                   # 7
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        # print(attention_mask.shape)         # torch.Size([224, 64])

        # 5. Encoder,
        # 输入:
        # (batch_size*n_channels, patch_num, d_model)
        # (batch_size*n_channels, patch_num)
        # 输出:
        # (batch_size*n_channels, patch_num, d_model)
        # print(type(self.encoder))           # <class 'transformers.models.t5.modeling_t5.T5Stack'>
        # print(enc_in.shape)                 # torch.Size([224, 64, 1024])
        # print(attention_mask.shape)         # torch.Size([224, 64])
        # outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        outputs = self.backbone_forward(inputs_embeds=enc_in, attention_mask=attention_mask)
        # print(type(outputs))                # <class 'transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions'>
        enc_out = outputs.last_hidden_state
        # print(enc_out.shape)                # torch.Size([224, 64, 1024])

        # Inverse-ChannelIndependence
        # (batch_size*n_channels, patch_num, d_model) -> (batch_size, n_channels, patch_num, d_model)
        # print(self.config.d_model)          # 1024
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model)).contiguous()
        # print(enc_out.shape)                # torch.Size([32, 7, 64, 1024])

        # 6. OutputHead & RevIN
        # 首先将patch_num和d_model维度拉直, 随后通过linear将其映射到输出长度
        # (batch_size, n_channels, patch_num, d_model) -> (batch_size, n_channels, patch_num*d_model)
        # (batch_size, n_channels, patch_num*d_model) -> (batch_size, n_channels, pred_len)
        # print(type(self.head))              # <class 'models.moment.ForecastingHead'>
        # print(type(self.normalizer))        # <class 'models.moment_module.RevIN'>
        # print(enc_out.shape)                # torch.Size([32, 7, 64, 1024])
        self.head.to(enc_out.device)
        dec_out = self.head(enc_out)
        # print(dec_out.shape)                # torch.Size([32, 7, 192])
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        # print(dec_out.shape)                # torch.Size([32, 7, 192])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 192, 7])
        # print('############# Long Forecasting-2')
        return dec_out

    def reconstruction(self, x_enc, input_mask, mask):
        # print('############# Reconstruction-1')
        batch_size, seq_len, n_channels = x_enc.shape

        # ChannelIndependence
        # print(x_enc.shape)                  # torch.Size([32, 512, 7])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 7, 512])
        # print(input_mask.shape)             # torch.Size([32, 512])
        x_enc = x_enc.reshape((-1, 1, seq_len)).contiguous()
        input_mask = input_mask.repeat_interleave(n_channels, axis=0)
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        # print(input_mask.shape)             # torch.Size([224, 512])

        # 1. 基于RevIN实现历史序列的【可逆实例规范化】
        # print(type(self.normalizer))        # <class 'models.moment_module.RevIN'>
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        # print(mask.shape)                   # torch.Size([224, 512])
        # print(input_mask.shape)             # torch.Size([224, 512])
        # print((mask*input_mask).shape)      # torch.Size([224, 512])
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])

        # 2. 基于一种无需训练的方式, 直接将长度为512的历史序列拆分为64个patch, 每个patch长度为8
        # print(type(self.tokenizer))         # <class 'models.moment_module.Patching'>
        # print(x_enc.shape)                  # torch.Size([224, 1, 512])
        x_enc = self.tokenizer(x=x_enc)
        # print(x_enc.shape)                  # torch.Size([224, 1, 64, 8])

        # 3. 将patch_len(8)映射为d_model(1024),
        # print(type(self.patch_embedding))   # <class 'models.moment_module.PatchEmbedding'>
        # print(x_enc.shape)                  # torch.Size([224, 1, 64, 8])
        # print(mask.shape)                   # torch.Size([224, 512])
        enc_in = self.patch_embedding(x_enc, mask=mask)
        # print(enc_in.shape)                 # torch.Size([224, 1, 64, 1024])
        # print(self.config.d_model)          # 1024
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model)).contiguous()
        # print(enc_in.shape)                 # torch.Size([224, 64, 1024])

        # 4. 为Encoder架构生成attention_mask, 其中每个样本的每个token对应一个值, 通道间共有mask值
        # print(input_mask.shape)             # torch.Size([224, 512])
        # print(self.patch_len)               # 8
        attention_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        # print(attention_mask.shape)         # torch.Size([224, 64])

        # 5. Encoder,
        # 输入:
        #   (batch_size*n_channels, patch_num, d_model)
        #   (batch_size*n_channels, patch_num)
        # 输出:
        #   (batch_size*n_channels, patch_num, d_model)
        # print(self.config.transformer_type) # encoder_only
        # print(type(self.encoder))           # <class 'transformers.models.t5.modeling_t5.T5Stack'>
        # print(enc_in.shape)                 # torch.Size([224, 64, 1024])
        # print(attention_mask.shape)         # torch.Size([224, 64])
        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(inputs_embeds=enc_in, decoder_inputs_embeds=enc_in, attention_mask=attention_mask)
        else:
            # outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
            outputs = self.backbone_forward(inputs_embeds=enc_in, attention_mask=attention_mask)
        # print(type(outputs))                # <class 'transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions'>
        enc_out = outputs.last_hidden_state
        # print(enc_out.shape)                # torch.Size([224, 64, 1024])
        # print(self.config.d_model)          # 1024
        enc_out = enc_out.reshape((-1, 1, n_patches, self.config.d_model)).contiguous()
        # print(enc_out.shape)                # torch.Size([224, 1, 64, 1024])

        # 6. OutputHead & RevIN
        # 首先基于linear将d_model映射回patch_len, 随后将其拉直为seq_len
        # (batch_size, n_channels, patch_num, d_model) -> (batch_size, n_channels, patch_num, patch_len)
        # (batch_size, n_channels, patch_num, patch_len) -> (batch_size, n_channels, seq_len)
        # print(type(self.head))              # <class 'models.moment.PretrainHead'>
        # print(type(self.normalizer))        # <class 'models.moment_module.RevIN'>
        # print(enc_out.shape)                # torch.Size([224, 1, 64, 1024])
        dec_out = self.head(enc_out)
        # print(dec_out.shape)                # torch.Size([224, 1, 512])
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        # print(dec_out.shape)                # torch.Size([224, 1, 512])

        # Inverse-ChannelIndependence
        # print(dec_out.shape)                # torch.Size([224, 1, 512])
        dec_out = dec_out.reshape((-1, n_channels, dec_out.shape[2])).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 7, 512])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print(dec_out.shape)                # torch.Size([32, 512, 7])
        # print('############# Reconstruction-2')
        return dec_out

    def classify(self, x_enc, input_mask):
        # print('############# Classification-1')
        batch_size, seq_len, n_channels = x_enc.shape

        # print(x_enc.shape)                  # torch.Size([32, 512, 1])
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # print(x_enc.shape)                  # torch.Size([32, 1, 512])
        # print(input_mask.shape)             # torch.Size([32, 512])

        # 2. 基于一种无需训练的方式, 直接将长度为512的历史序列拆分为64个patch, 每个patch长度为8
        # print(type(self.tokenizer))         # <class 'models.moment_module.Patching'>
        # print(type(self.patch_embedding))   # <class 'models.moment_module.PatchEmbedding'>
        # print(x_enc.shape)                  # torch.Size([32, 1, 512])
        x_enc = self.tokenizer(x=x_enc)
        # print(x_enc.shape)                  # torch.Size([32, 1, 64, 8])

        # 3. 将patch_len(8)映射为d_model(1024),
        # print(input_mask.shape)             # torch.Size([32, 512])
        enc_in = self.patch_embedding(x_enc, mask=input_mask)
        # print(enc_in.shape)                 # torch.Size([32, 1, 64, 1024])

        # ChannelIndependence
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape((batch_size * n_channels, n_patches, self.config.d_model)).contiguous()
        # print(enc_in.shape)                 # torch.Size([32, 64, 1024])

        # 4. 为Encoder架构生成attention_mask, 其中每个样本的每个token对应一个值, 通道间共有mask值
        # print(input_mask.shape)             # torch.Size([32, 512])
        # print(self.patch_len)               # 8
        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        # print(patch_view_mask.shape)        # torch.Size([32, 64])
        # print(n_channels)                   # 1
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        # print(attention_mask.shape)         # torch.Size([32, 64])

        # 5. Encoder,
        # print(type(self.encoder))           # <class 'transformers.models.t5.modeling_t5.T5Stack'>
        # print(enc_in.shape)                 # torch.Size([32, 64, 1024])
        # print(attention_mask.shape)         # torch.Size([32, 64])
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        # outputs = self.backbone_forward(inputs_embeds=enc_in, attention_mask=attention_mask)
        # print(type(outputs))                # <class 'transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions'>
        enc_out = outputs.last_hidden_state
        # print(enc_out.shape)                # torch.Size([32, 64, 1024])

        # Inverse-ChannelIndependence
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model)).contiguous()
        # print(enc_out.shape)                # torch.Size([32, 1, 64, 1024])

        # 6. ChannelFusion
        #   当reduction为mean,则计算全部C个通道的均值,
        #   当reduction为concat,则将通道数量C和d_model拉直
        reduction = "concat"
        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)
        elif reduction == "concat":
            enc_out = enc_out.permute(0, 2, 3, 1).contiguous().reshape(batch_size, n_patches, self.config.d_model * n_channels).contiguous()
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")
        # print(enc_out.shape)                # torch.Size([32, 64, 1024])

        # 7. OutputHead
        # 首先对patch_num维度求均值, 随后通过linear将d_model/n_channels*d_model映射到num_class
        # (batch_size, patch_num, d_model) -> (batch_size, d_model) -> (batch_size, num_class)
        # (batch_size, patch_num, n_channels*d_model) -> (batch_size, n_channels*d_model) -> (batch_size, num_class)
        # print(type(self.head))              # <class 'models.moment.ClassificationHead'>
        self.head.to(enc_out.device)
        # print(enc_out.shape)                # torch.Size([32, 64, 1024])
        logits = self.head(enc_out)
        # print(logits.shape)                 # torch.Size([32, 5])

        # print('############# Classification-2')
        return logits, enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).float()
            # print(input_mask.shape)             # torch.Size([32, 512])
            output = self.forecast(x_enc=x_enc, input_mask=input_mask)
            # print(output.shape)                 # torch.Size([32, 192, 7])
            return output

        elif self.task_name == 'imputation':
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).long()
            # print(input_mask.shape)             # torch.Size([32, 512])
            # print(mask.shape)                   # torch.Size([32, 512, 7])
            mask = mask.permute(0, 2, 1).contiguous()
            # print(mask.shape)                   # torch.Size([32, 7, 512])
            mask = mask.reshape((-1, mask.shape[-1])).long().contiguous()
            # print(mask.shape)                   # torch.Size([224, 512])
            output = self.reconstruction(x_enc=x_enc, input_mask=input_mask, mask=mask)
            # print(output.shape)                 # torch.Size([32, 512, 7])
            return output

        elif self.task_name == 'anomaly_detection':
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).long()
            # print(input_mask.shape)             # torch.Size([32, 512])
            # print(mask.shape)                   # torch.Size([32, 512, 7])
            mask = mask.permute(0, 2, 1).contiguous()
            # print(mask.shape)                   # torch.Size([32, 7, 512])
            mask = mask.reshape((-1, mask.shape[-1])).long().contiguous()
            # print(mask.shape)                   # torch.Size([224, 512])
            output = self.reconstruction(x_enc=x_enc, input_mask=input_mask, mask=mask)
            # print(output.shape)                 # torch.Size([32, 512, 7])
            return output

        # 在分类任务中，输入的时间序列样本往往不到512个时间步，因此需要在原始序列的前面填充零，
        # 与之对应地，input_mask为0代表是填充值、为1代表是真实值，因此对于预测插补异常检测的input_mask为全1张量
        elif self.task_name == 'classification':
            # 填充x_enc
            # print(x_enc.shape)                  # torch.Size([32, 140, 7])
            orig_len = x_enc.shape[1]
            # print(orig_len)                     # 140
            x_enc = torch.cat([
                torch.zeros((x_enc.shape[0], 512 - x_enc.shape[1], x_enc.shape[2])).to(x_enc.device).float(),
                x_enc,
            ], dim=1).to(x_enc.device).float()
            # print(x_enc.shape)                  # torch.Size([32, 512, 7])
            # 填充input_mask
            input_mask = torch.ones((x_enc.shape[0], x_enc.shape[1])).to(x_enc.device).long()
            input_mask[:, :(x_enc.shape[1] - orig_len)] = 0
            # print(input_mask.shape)             # torch.Size([32, 512])
            logits, embedding = self.classify(x_enc=x_enc, input_mask=input_mask)
            # print(logits.shape)                 # torch.Size([32, 5])
            # print(embedding.shape)              # torch.Size([32, 64, 896])
            return logits, embedding

        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class ModelPipeline(Model, PyTorchModelHubMixin):
    def __init__(self, config, **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        task_name = kwargs.get("model_kwargs", {}).pop("task_name")
        pred_len = kwargs.get("model_kwargs", {}).pop("pred_len", None)
        n_channels = kwargs.get("model_kwargs", {}).pop("n_channels", None)
        num_class = kwargs.get("model_kwargs", {}).pop("num_class", None)
        super().__init__(config=config, adapter_task_name=task_name, pred_len=pred_len, n_channels=n_channels, num_class=num_class, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == 'long_term_forecast' or config.task_name == 'short_term_forecast':
            if not hasattr(config, "pred_len"):
                raise ValueError("pred_len must be specified for long-horizon forecasting.")

        if config.task_name == 'classification':
            if not hasattr(config, "n_channels"):
                raise ValueError("n_channels must be specified for classification.")
            if not hasattr(config, "num_class"):
                raise ValueError("num_class must be specified for classification.")

    def init(self) -> None:
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'classification']:
            self.head = self._get_head(self.task_name)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_enc = torch.zeros((2, 512, 7)).to(device).float()

    # Forecasting
    model = ModelPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'long_term_forecast', 'pred_len': 192},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    print(x_enc.shape)                  # torch.Size([32, 512, 7])
    output_forecasting = model(x_enc=x_enc)
    print(output_forecasting.shape)     # torch.Size([32, 192, 7])

    # Imputation
    model = ModelPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'imputation'},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    mask = torch.rand(x_enc.shape).to(x_enc.device)
    mask[mask <= 0.3] = 0
    mask[mask > 0.3] = 1
    print(x_enc.shape)                  # torch.Size([32, 512, 7])
    print(mask.shape)                   # torch.Size([32, 512, 7])
    output_imputation = model(x_enc=x_enc, mask=mask)
    print(output_imputation.shape)      # torch.Size([32, 512, 7])

    # Anomaly_Detection
    model = ModelPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "anomaly_detection"},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    mask = torch.rand(x_enc.shape).to(x_enc.device).long()
    mask[mask <= 0.3] = 0
    mask[mask > 0.3] = 1
    print(x_enc.shape)                  # torch.Size([32, 512, 7])
    print(mask.shape)                   # torch.Size([32, 512, 7])
    output_anomaly = model(x_enc=x_enc, mask=mask)
    print(output_anomaly.shape)         # torch.Size([32, 512, 7])

    # Classification
    # 在分类任务中，输入的时间序列样本往往不到512个时间步，因此需要在原始序列的前面填充零，
    # 与之对应地，input_mask为0代表是填充值、为1代表是真实值，因此对于预测插补异常检测的input_mask为全1张量
    x_enc = torch.zeros((2, 140, 7)).to(device).float()
    model = ModelPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={'task_name': 'classification', 'n_channels': 7, 'num_class': 5},
        local_files_only=True,
    ).to(device).float()
    if hasattr(model, 'init'):
        print('With Init')
        model.init()
    else:
        print('Without Init')
    print(x_enc.shape)                  # torch.Size([32, 140, 7])
    logits, embedding = model(x_enc=x_enc)
    print(logits.shape)                 # torch.Size([32, 5])
    print(embedding.shape)              # torch.Size([32, 64, 896])
