model_name=PatchTST_Adapter
d_model=16		# 64
d_ff=32			# 128
n_heads=4		# 8
e_layers=4
batch_size=16

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 0 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 0 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 0 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 0 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 0 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --loss 'SMAPE'

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers $e_layers \
  --d_layers 0 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1 \
  --loss 'SMAPE'
