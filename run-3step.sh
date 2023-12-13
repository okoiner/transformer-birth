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
