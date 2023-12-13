# 1) loss_head_only, no stepped training, no ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=False

# 2) loss_head_only, no stepped training, no ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=False

# 3) loss_head_only, no stepped training, ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=False

# 4) loss_head_only, no stepped training, ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=False

# 5) loss_head_only, stepped training, no ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=True

# 6) loss_head_only, stepped training, no ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=True

# 7) loss_head_only, stepped training, ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=True

# 8) loss_head_only, stepped training, ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=True \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=True

# 9) global loss, no stepped training, no ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=False

# 10) global loss, no stepped training, no ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=False

# 11) global loss, no stepped training, ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=False

# 12) global loss, no stepped training, ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=False

# 13) loss_head_only, stepped training, no ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=True

# 14) loss_head_only, stepped training, no ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=False \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=True

# 15) loss_head_only, stepped training, ff, fixed trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=True\
  train_stepped=True

# 16) loss_head_only, stepped training, ff, random trigger
python3 ihead_basic_3step_main.py \
  max_iters=1000 \
  log_probes=True \
  optim_args.use_sgd=True \
  model_args.final_ffn=True \
  optim_args.weight_decay=1e-4 \
  optim_args.batch_size=512 \
  eval_delta=5 \
  model_args.freeze_embeddings=True \
  model_args.freeze_output=True \
  model_args.tie_output=False \
  loss_head_only=False \
  data_args.k=5 \
  model_args.dim=128 \
  optim_args.learning_rate=0.5 \
  data_args.fixed_special_toks=False\
  train_stepped=True
