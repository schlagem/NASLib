%%bash
optimizers=(bananas re rs)
predictors=(mlp lgb xgb rf bayes_lin_reg gp)
constraints=(latency parameters)

start_seed=0

# search space / data:
search_space=ofa
dataset=imagenet
search_epochs=300

# trials / seeds:
trials=3

# create config files
for i in $(seq 0 $((${#optimizers[@]}-1)))
do
  for j in $(seq 0 $((${#predictors[@]}-1)))
  do
    for k in $(seq 0 $((${#constraints[@]}-1)))
    do
      optimizer=${optimizers[$i]}
      predictor=${predictors[$j]}
      constraint=${constraints[$k]}
      out_dir=docs/$optimizer\_run
      python create_configs.py --predictor $predictor \
      --epochs $search_epochs --start_seed $start_seed --trials $trials \
      --out_dir $out_dir --dataset=$dataset --config_type nas_predictor_constraint \
      --search_space $search_space --optimizer $optimizer --constraint $constraint
    done
  done
done