model_name=DLinear_Adapter
seq_len=336
d_model=32		# 64
d_ff=64			# 128
n_heads=4		# 8
e_layers=8
batch_size=128

python -u run_for_ZeroShot.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_96 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 96 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1

python -u run_for_ZeroShot.py \
   --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_192 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 192 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1

python -u run_for_ZeroShot.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_336 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 336 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1

python -u run_for_ZeroShot.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len\_720 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data ETTh2 \
  --features M \
  --label_len 0 \
  --pred_len 720 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1