model_name=TimeMixer_Adapter
d_model=8
d_ff=16
e_layers=4
batch_size=128

down_sampling_layers=3
down_sampling_window=2

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SWaT \
  --model_id SWAT \
  --data SWAT \
  --model $model_name \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --enc_in 51 \
  --c_out 51 \
  --des 'Exp' \
  --anomaly_ratio 1 \
  --learning_rate 0.00002 \
  --itr 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
