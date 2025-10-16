model_name=DLinear
seq_len=336
batch_size=32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_96 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_192 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 192 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_336 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_720 \
  --model $model_name \
  --seq_len $seq_len \
  --batch_size $batch_size \
  --data Solar \
  --features M \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0002 \
  --itr 1
