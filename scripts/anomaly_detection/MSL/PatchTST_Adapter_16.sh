model_name=PatchTST_Adapter
d_model=16		# 64
d_ff=32			# 128
n_heads=4		# 8
e_layers=4
batch_size=128

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --data MSL \
  --model $model_name \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --factor 3 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --learning_rate 0.0001 \
  --train_epochs 40 \
  --itr 1