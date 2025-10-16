model_name=PatchTST
seq_len=336
d_model=64
d_ff=128
e_layers=4
batch_size=32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_96 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_192 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model 32 \
  --d_ff 64 \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 192 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_336 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model 16 \
  --d_ff 32 \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_720 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1
