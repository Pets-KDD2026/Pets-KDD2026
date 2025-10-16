export CUDA_VISIBLE_DEVICES=0

########################## best results

bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTh1_32.sh
bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTh2_16.sh
bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTm1_16.sh
bash ./scripts/long_term_forecast/ETT_script/PatchTST_Adapter/ETTm2_16.sh
bash ./scripts/long_term_forecast/Weather_script/PatchTST_Adapter/Weather_16.sh
bash ./scripts/long_term_forecast/Solar_script/PatchTST_Adapter.sh
bash ./scripts/long_term_forecast/ECL_script/PatchTST_Adapter/ECL_32.sh
bash ./scripts/long_term_forecast/Traffic_script/PatchTST_Adapter
bash ./scripts/short_term_forecast/TimeMixer_Adapter_PEMS.sh

bash ./scripts/anomaly_detection/SMD/TimeMixer_Adapter.sh
bash ./scripts/anomaly_detection/MSL/TimeMixer_Adapter.sh
bash ./scripts/anomaly_detection/PSM/TimeMixer_Adapter.sh
bash ./scripts/anomaly_detection/SMAP/TimeMixer_Adapter.sh
bash ./scripts/anomaly_detection/SWAT/TimeMixer_Adapter.sh
