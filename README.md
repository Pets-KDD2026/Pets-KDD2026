# Pets (Submitted to ICLR 2025)

### This is an offical implementation of "Pets: Generalizable Pattern Assisted Architecture for Time Series Analysis" submitted to NIPS2025.

[//]: # (:triangular_flag_on_post: Our model has been included in [GluonTS]&#40;https://github.com/awslabs/gluonts&#41;. Special thanks to the contributor @[kashif]&#40;https://github.com/kashif&#41;!)

[//]: # (:triangular_flag_on_post: Our model has been included in [NeuralForecast]&#40;https://github.com/Nixtla/neuralforecast&#41;. Special thanks to the contributor @[kdgutier]&#40;https://github.com/kdgutier&#41; and @[cchallu]&#40;https://github.com/cchallu&#41;!)

[//]: # (:triangular_flag_on_post: Our model has been included in [timeseriesAI&#40;tsai&#41;]&#40;https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb&#41;. Special thanks to the contributor @[oguiza]&#40;https://github.com/oguiza&#41;!)


This paper presents a novel architecture capable of adapting to various time-series analysis tasks. Such a universal pattern recognition strategy can enhance the performance of existing methods.

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/pic1.png)

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/pic2.png)

## Results

### Overview

:star2: **one-model many-tasks**: Pets achieves state-of-the-art performances in all 8 mainstream time series analysis tasks across 60 benchmarks

:star2: **plug-and-play augmentation strategy**: Pets can consistently enhance the performance of disparate model architectures on extensive datasets and tasks.

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/pic3.png)

The proposed Spectrum Decomposition and Amplitude Quantization (SDAQ) and Fluctuation Pattern Assisted (FPA) strategy can be seamlessly integrated into various deep models, in a plug-and-play fashion. This prompts us to investigate its generality by inserting FPA into diverse types of structures. We selected representative baselines composed of divergent underlying architectures. 

Concretely, the attention-based PatchTST witnessed performance improvements of **34.8%** and **17.6%** on Solar and ECL respectively. The Linear-based TimeMixer and DLinear achieved enhancements of **14.5%** and **19.4%** on Traffic. The CNN-based TimesNet showed an improvement of **8.8%** on ETTm2. These findings authenticate that the proposed augmentation strategy is applicable to variegated deep architectures.

### Long-term Forecasting

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/table1.png)

Pets outperforms other models in long-term forecasting across various datasets. The MSE of Pets is reduced by **8.7%** and **15.1%** compared to the TimeMixer and iTransformer, respectively. For ETT (Avg), Pets achieves **7.4%** lower MSE than TimeMixer. Specifically, it outperforms the runner-up model by a margin of **13.4%** and **19.7%** on the challenging Solar-Energy.

### Short-term Forecasting

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/table2.png)

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/table3.png)

In the M4 dataset, compared to the challenging TimeMixer and TimesNet, Pets accomplishes a reduction of **5.6%** and **7.2%** in MASE, respectively. Within the PEMS, in comparison to the leading TimeMixer and iTransformer, Pets reduces MAPE by **5.7%** and **20.3%**. Remarkably, in comparison with the conventional PatchTST, Pets exhibits performance augmentations on M4 and PEMS that exceed **24.4%** and **33.1%**.

### Anomaly Detection

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/table4.png)

### Classification

![alt text](https://github.com/sa13d3asda/Pets-NIPS2025/blob/main/pic/table5.png)

## Getting Started

1. Install requirements. 

```
pip install -r requirements.txt
```

3. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

4. Training. All the scripts are in the directory ```./scripts```. In the experimental results presented in the manuscript, we choose **transformer-based Pets** as the default setting, and specifically, we refer to such a model as **PatchTST-Adapter**. Similarly, we also provide **TimeMixer-Adapter**, **DLinear-Adapter**, and **TimesNet-Adapter** to represent **Linear and convolutional based Pets**, respectively. For example, if you want to get the multivariate forecasting results for **ETT** dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:

```
bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTh1_32.sh

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTh2_16.sh

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTm1_16.sh

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTm2_16.sh
```

In addition, we provide scripts for anomaly detection and zero-shot inference.

```
bash ./scripts/anomaly_detection/SMD/TimeMixer_Adapter.sh

bash ./scripts/anomaly_detection/SMD/TimesNet_Adapter.sh

bash ./scripts/anomaly_detection/SMD/PatchTST_Adapter_8.sh

bash ./scripts/anomaly_detection/SMD/PatchTST_Adapter_16.sh

bash ./scripts/anomaly_detection/SMD/PatchTST_Adapter_32.sh
```

and

```
bash ./scripts/zero_shot/PatchTST_Adapter/ETTh1_32.sh

bash ./scripts/zero_shot/PatchTST_Adapter/ETTh2_16.sh

bash ./scripts/zero_shot/PatchTST_Adapter/ETTm1_16.sh

bash ./scripts/zero_shot/PatchTST_Adapter/ETTm2_16.sh
```

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and backbone modules.). We also provide codes for the baseline models.

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai

## Contact

## Citation

