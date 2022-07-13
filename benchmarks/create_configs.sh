%%bash
optimizer=bananas
predictors=(mlp lgb xgb rf bayes_lin_reg gp)

start_seed=0

# folders:
# this supposes your location is at NASLib/docs. Change the base_file location based on where you
# opened the notebook
base_file=../benchmarks
save_dir=$optimizer\_run
out_dir=docs/$save_dir\_$start_seed

# search space / data:
search_space=ofa
dataset=imagenet
search_epochs=100

# trials / seeds:
trials=3
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    python $base_file/create_configs.py --predictor $predictor \
    --epochs $search_epochs --start_seed $start_seed --trials $trials \
    --out_dir $out_dir --dataset=$dataset --config_type nas_predictor \
    --search_space $search_space --optimizer $optimizer
done