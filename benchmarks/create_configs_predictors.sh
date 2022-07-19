%%bash
optimizers=(bananas re rs)
predictors=(mlp lgb xgb rf bayes_lin_reg gp none)
constraints=(latency parameters)
# here the max size of our net is 29.239MB and min size 13.011MB: These are 3 quartiles
parameters_constraint=(17.068038940429688 21.12493896484375 25.181838989257812)
# TODO
latency_constraint=(25 50 75)

start_seed=0

# search space / data:
search_space=ofa
dataset=imagenet
search_epochs=300

# trials / seeds:
# Here used as quartiles: 25,50,75,100
trials=1

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
      if [ $constraint = parameters ]
        then
          efficiency=${parameters_constraint[*]}
        else
          efficiency=${latency_constraint[*]}
      fi
      python create_configs.py --predictor $predictor \
      --epochs $search_epochs --start_seed $start_seed --trials $trials \
      --out_dir $out_dir --dataset=$dataset --config_type nas_predictor_constraint \
      --search_space $search_space --optimizer $optimizer --constraint $constraint --efficiency $efficiency
    done
  done
done

for i in $(seq 0 $((${#optimizers[@]}-1)))
do
  for j in $(seq 0 $((${#predictors[@]}-1)))
  do
    optimizer=${optimizers[$i]}
    predictor=${predictors[$j]}
    out_dir=docs/$optimizer\_run
    python create_configs.py --predictor $predictor \
    --epochs $search_epochs --start_seed $start_seed --trials $trials \
    --out_dir $out_dir --dataset=$dataset --config_type nas_predictor \
    --search_space $search_space --optimizer $optimizer
  done
done