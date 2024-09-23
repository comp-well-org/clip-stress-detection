for fold in 3; do
    for seed in 2 3; do
        python gen_main1.py --dataset pmdata --seed $seed --fold $fold --id
    done
done