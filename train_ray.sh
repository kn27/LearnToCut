set -e

for seed in 1 
do
for filter in MeanStdFilter NoFilter
do
for n_directions in 11
do
for step in  0.1 0.2 0.05
do
for std in 0.02 0.05 0.1
do
for type in attention #attentionembed lstmembed 
do

python main.py --config custom_config --policy_type $type --n_iter 80 --n_directions $n_directions --seed $seed --step_size $step --delta_std $std --n_workers 12 --rollout_length 50 --tag main --deltas_used 8

done
done
done
done
done
done
