# Pets

Official code for "Pets: Generalizable Pattern Assisted Architecture for Time Series Analysis" submitted to NIPS2025.

## Install

To ensure fair experimental comparison, we adopted the same experimental environment and data preprocessing methods as TimesNet, with more details available in its project.

## Run

You can execute the following commands to quickly run our code:

### Forecasting

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTh1_32.sh

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTh2_16.sh

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTm1_16.sh

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTm2_16.sh

### Anomaly Detection

bash ./scripts/anomaly_detection/SMD/PatchTST_Adapter_8.sh

bash ./scripts/anomaly_detection/SMD/PatchTST_Adapter_16.sh

bash ./scripts/anomaly_detection/SMD/PatchTST_Adapter_32.sh

bash ./scripts/anomaly_detection/SMD/TimeMixer_Adapter.sh

bash ./scripts/anomaly_detection/SMD/TimesNet_Adapter.sh

### Zero-shot Forecasting

bash ./scripts/zero_shot/PatchTST_Adapter/ETTh1_32.sh

bash ./scripts/zero_shot/PatchTST_Adapter/ETTh2_16.sh

bash ./scripts/zero_shot/PatchTST_Adapter/ETTm1_16.sh

bash ./scripts/zero_shot/PatchTST_Adapter/ETTm2_16.sh


