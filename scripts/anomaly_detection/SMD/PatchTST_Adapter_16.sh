model_name=PatchTST_Adapter
d_model=16		# 64
d_ff=32			# 128
n_heads=4		# 8
e_layers=4
batch_size=128

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --data SMD \
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
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --learning_rate 0.0002 \
  --train_epochs 40 \
  --itr 1