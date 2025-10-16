model_name=TimesNet_Adapter
seq_len=336
d_model=64		# 64
d_ff=128			# 128
n_heads=8		# 8
e_layers=12
batch_size=32

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTm2 \
  --features M \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTm2 \
  --features M \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTm2 \
  --features M \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTm2 \
  --features M \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1
