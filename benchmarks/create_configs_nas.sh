%%bash
optimizers=(rs re)

start_seed=0

# search space / data:
search_space=ofa
dataset=imagenet
search_epochs=100

# trials / seeds:
trials=3

# create config files
for i in $(seq 0 $((${#optimizers[@]}-1)) )
do
    optimizer=${optimizers[$i]}
    out_dir=docs/$optimizer\_run
    python create_configs.py \
    --epochs $search_epochs --start_seed $start_seed --trials $trials \
    --out_dir $out_dir --dataset=$dataset --config_type nas \
    --search_space $search_space --optimizer $optimizer
done