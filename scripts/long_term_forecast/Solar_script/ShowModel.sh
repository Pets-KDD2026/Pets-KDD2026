model_name=ShowModel
seq_len=336

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len\_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1