model_name=TimesNet_Adapter
seq_len=336
d_model=32
d_ff=64
n_heads=4
e_layers=4
batch_size=32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len\_96 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 96 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len\_192 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 192 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len\_336 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 336 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len\_720 \
  --model $model_name \
  --seq_len $seq_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --data custom \
  --features M \
  --label_len 0 \
  --pred_len 720 \
  --d_layers 0 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1